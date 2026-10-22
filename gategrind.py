"""A bounded crawler for publicly accessible GATE Grind question pages.

It intentionally follows only the site's paper and subject routes.  This is not
a general purpose site scraper: it is rate limited, page limited, checks
robots.txt when available, and keeps a local SQLite copy of extracted text.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from config import DATA_DIR

DOMAIN = "thegategrind.online"
CATALOG_URL = f"https://{DOMAIN}/pyq"
USER_AGENT = "personal-rag-gategrind-crawler/1.0 (educational RAG; respects robots.txt)"
PAGE_PATH = re.compile(r"^/(?:exam|subject)/\d+/?$")
MAX_PAGES = 100
MAX_RESPONSE_BYTES = 2_000_000
CACHE_DB = DATA_DIR / "gategrind_cache.sqlite3"


@dataclass
class CrawlResult:
    documents: list[Document]
    pages_seen: int
    pages_changed: int
    cached_pages: int
    errors: list[str]


def _normalise(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(("https", parsed.netloc.lower(), parsed.path.rstrip("/") or "/", "", "", ""))


def _is_allowed(url: str, *, allow_catalog: bool = False) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != DOMAIN:
        return False
    return (allow_catalog and parsed.path.rstrip("/") == "/pyq") or bool(PAGE_PATH.fullmatch(parsed.path))


def _open_cache() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
            content_hash TEXT NOT NULL, fetched_at TEXT NOT NULL
        )"""
    )
    return conn


def cached_documents() -> list[Document]:
    """Return locally preserved GATE Grind pages for (re)indexing without a crawl."""
    if not CACHE_DB.exists():
        return []
    with _open_cache() as cache:
        rows = cache.execute("SELECT url, title, content FROM pages ORDER BY fetched_at").fetchall()
    return [Document(
        page_content=content,
        metadata={"source": f"Gate Grind: {title}", "url": url, "site": "thegategrind.online"},
    ) for url, title, content in rows]


def _robots(session: requests.Session) -> RobotFileParser | None:
    response = session.get(f"https://{DOMAIN}/robots.txt", timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser


def _extract(html: str, url: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    root = soup.select_one("main, article, [role='main']") or soup.body or soup
    for tag in root.select("script, style, noscript, nav, footer, header, aside, form"):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", root.get_text("\n", strip=True)).strip()
    links = [_normalise(urljoin(url, a["href"])) for a in soup.select("a[href]")]
    return title, text, links


def crawl(
    seed_url: str = CATALOG_URL,
    *,
    max_pages: int = 25,
    delay_seconds: float = 1.0,
    progress: Callable[[str], None] | None = None,
    session: requests.Session | None = None,
) -> CrawlResult:
    """Fetch a small, explicit portion of The GATE Grind and cache page text.

    The catalog is used only to discover paper/subject URLs; it is not indexed.
    Passing an individual ``/exam/<id>`` or ``/subject/<id>`` URL fetches just
    that page.  A page whose cached text has not changed is not re-indexed.
    """
    if not 1 <= max_pages <= MAX_PAGES:
        raise ValueError(f"max_pages must be between 1 and {MAX_PAGES}")
    if delay_seconds < 0.5:
        raise ValueError("delay_seconds must be at least 0.5 seconds")
    seed_url = _normalise(seed_url)
    if not _is_allowed(seed_url, allow_catalog=True):
        raise ValueError(f"Use https://{DOMAIN}/pyq or a specific /exam/<id> or /subject/<id> URL.")

    own_session = session is None
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    errors: list[str] = []
    documents: list[Document] = []
    cached_pages = 0
    pages_seen = 0
    queue = [seed_url]
    visited: set[str] = set()

    try:
        try:
            robots = _robots(session)
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not check robots.txt; crawl stopped: {exc}") from exc
        with _open_cache() as cache:
            while queue and pages_seen < max_pages:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)
                if robots and not robots.can_fetch(USER_AGENT, url):
                    errors.append(f"Blocked by robots.txt: {url}")
                    continue
                if progress:
                    progress(f"Fetching {pages_seen + 1}/{max_pages}: {url}")
                try:
                    response = session.get(url, timeout=25)
                    response.raise_for_status()
                    if len(response.content) > MAX_RESPONSE_BYTES:
                        raise ValueError("response exceeds 2 MB safety limit")
                    title, text, links = _extract(response.text, url)
                    if not text:
                        raise ValueError("no readable page text found")
                except (requests.RequestException, ValueError) as exc:
                    errors.append(f"{url}: {exc}")
                    time.sleep(delay_seconds)
                    continue

                # A catalog page discovers permitted pages but is not itself useful RAG content.
                if url.rstrip("/") == CATALOG_URL:
                    queue.extend(link for link in links if _is_allowed(link) and link not in visited)
                else:
                    pages_seen += 1
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    previous = cache.execute("SELECT content_hash FROM pages WHERE url = ?", (url,)).fetchone()
                    cache.execute(
                        "INSERT OR REPLACE INTO pages(url, title, content, content_hash, fetched_at) VALUES (?, ?, ?, ?, ?)",
                        (url, title, text, digest, datetime.now(timezone.utc).isoformat()),
                    )
                    if previous and previous[0] == digest:
                        cached_pages += 1
                    else:
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": f"Gate Grind: {title}", "url": url, "site": "thegategrind.online"},
                        ))
                cache.commit()
                time.sleep(delay_seconds)
    finally:
        if own_session:
            session.close()
    return CrawlResult(documents, pages_seen, len(documents), cached_pages, errors)
