"""Routing a model string to LiteLLM call kwargs (no LLM calls)."""

import warnings

import pytest

from cafe import llm


@pytest.fixture
def cloud_key(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "key")
    monkeypatch.setattr(llm, "_ollama_host_warned", set(), raising=False)


@pytest.mark.parametrize("host", [
    None,                        # unset: Ollama Cloud
    "https://ollama.com",        # the value .env.example used to set
    "https://ollama.com/",
    "https://ollama.com/v1",     # the OpenAI-compatible base pasted whole: was ".../v1/v1"
    "https://ollama.com/api/",   # Ollama's native API path
    " https://ollama.com ",      # stray whitespace: was a base URL without a scheme
])
def test_ollama_cloud_goes_to_ollama_com(monkeypatch, cloud_key, host):
    if host is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", host)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call = llm._route("ollama_cloud/gpt-oss:20b", {"temperature": 0.0})
    assert call == {"model": "openai/gpt-oss:20b", "temperature": 0.0,
                    "api_base": "https://ollama.com/v1", "api_key": "key"}


def test_ollama_cloud_goes_to_another_https_endpoint(monkeypatch, cloud_key):
    monkeypatch.setenv("OLLAMA_HOST", "https://gateway.example.com/ollama/v1")
    assert llm._route("ollama_cloud/m", {})["api_base"] == "https://gateway.example.com/ollama/v1"


@pytest.mark.parametrize("host", [
    "127.0.0.1:11434", "0.0.0.0", "localhost:11434",   # Ollama's own form for a server
    "http://localhost:11434", "http://gpu-box:11434",
])
def test_ollama_server_address_is_not_used_for_ollama_cloud(monkeypatch, cloud_key, host):
    """Ollama's tools read OLLAMA_HOST as the address of an Ollama server, and a shell export
    wins over .env: such a value gave a base URL without a scheme, or sent ollama_cloud/
    calls (with the cloud key) to that server."""
    monkeypatch.setenv("OLLAMA_HOST", host)
    with pytest.warns(UserWarning, match="not an https:// URL"):
        assert llm._route("ollama_cloud/m", {})["api_base"] == "https://ollama.com/v1"
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # warned once per value, not on every call
        assert llm._route("ollama_cloud/m", {})["api_base"] == "https://ollama.com/v1"
        # other models never read it: "ollama/" is LiteLLM's local daemon (OLLAMA_API_BASE)
        assert llm._route("ollama/llama3", {}) == {"model": "ollama/llama3"}
        assert llm._route("gpt-4o", {}) == {"model": "gpt-4o"}
