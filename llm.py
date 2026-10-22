"""LangChain chat models: local Ollama or Gemini. Both share the same interface."""
import base64
import os

from langchain_core.messages import HumanMessage

from config import GEMINI_MODEL, OLLAMA_MODEL


def get_llm(provider, gemini_key=None, temperature=0.2):
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = gemini_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API key missing. Add it in the sidebar or set GEMINI_API_KEY.")
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=key, temperature=temperature)


def describe_image(data, mime, gemini_key=None):
    """Multimodal step: Gemini turns an image into searchable text."""
    b64 = base64.b64encode(data).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": "Describe this image in detail. Transcribe any visible text and "
                                 "explain any charts, diagrams or code."},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ])
    return get_llm("gemini", gemini_key).invoke([msg]).content
