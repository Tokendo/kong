"""Tests for endpoint introspection.

The payloads below are what llama-server returns from /v1/models and /props.
Field names have moved between builds and differ from one OpenAI-compatible
server to the next, so the parser is expected to take what it recognizes and
leave the rest unknown.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from kong.llm.endpoint import (
    EndpointError,
    EndpointInfo,
    EndpointModel,
    discover,
    suggest_limits,
)

MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "kong-local",
            "object": "model",
            "created": 1_700_000_000,
            "owned_by": "llamacpp",
            "meta": {
                "vocab_type": 2,
                "n_vocab": 152064,
                "n_ctx_train": 32768,
                "n_embd": 3584,
                "n_params": 7_615_616_512,
                "size": 5_500_000_000,
            },
        }
    ],
}

PROPS_PAYLOAD = {
    "default_generation_settings": {"n_ctx": 16384, "temperature": 0.8},
    "total_slots": 1,
    "model_path": "/models/qwen2.5-coder-7b-instruct-q5_k_m.gguf",
    "build_info": "b4321",
}


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


@pytest.fixture
def server(monkeypatch):
    """Serve canned payloads by URL suffix, recording what was requested."""
    requested: list[str] = []
    headers: list[dict[str, str]] = []

    def install(routes: dict[str, object]):
        def fake_urlopen(request, timeout=None):
            requested.append(request.full_url)
            headers.append(dict(request.header_items()))
            for suffix, payload in routes.items():
                if request.full_url.endswith(suffix):
                    if isinstance(payload, Exception):
                        raise payload
                    body = payload if isinstance(payload, str) else json.dumps(payload)
                    return _FakeResponse(body.encode("utf-8"))
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {}, None
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return requested, headers

    return install


class TestDiscovery:
    def test_reads_models_and_runtime_context(self, server):
        server({"/models": MODELS_PAYLOAD, "/props": PROPS_PAYLOAD})

        info = discover("http://127.0.0.1:8080/v1")

        assert [m.id for m in info.models] == ["kong-local"]
        assert info.models[0].train_context == 32768
        assert info.models[0].parameters == 7_615_616_512
        assert info.context_tokens == 16384
        assert info.total_slots == 1
        assert info.build_info == "b4321"
        assert info.model_path.endswith(".gguf")

    def test_the_runtime_context_wins_over_the_training_context(self, server):
        """A server started with -c 16384 gives 16384, whatever the model trained on."""
        server({"/models": MODELS_PAYLOAD, "/props": PROPS_PAYLOAD})

        assert discover("http://127.0.0.1:8080/v1").effective_context == 16384

    def test_props_context_can_be_nested_under_params(self, server):
        server(
            {
                "/models": MODELS_PAYLOAD,
                "/props": {"default_generation_settings": {"params": {"n_ctx": 8192}}},
            }
        )

        assert discover("http://127.0.0.1:8080/v1").context_tokens == 8192

    def test_props_is_queried_beside_the_openai_surface(self, server):
        requested, _ = server({"/models": MODELS_PAYLOAD, "/props": PROPS_PAYLOAD})

        discover("http://127.0.0.1:8080/v1")

        assert "http://127.0.0.1:8080/v1/models" in requested
        assert "http://127.0.0.1:8080/props" in requested  # not /v1/props

    def test_an_api_key_is_sent_as_a_bearer_token(self, server):
        _, headers = server({"/models": MODELS_PAYLOAD, "/props": PROPS_PAYLOAD})

        discover("http://127.0.0.1:8080/v1", api_key="secret")

        assert any(h.get("Authorization") == "Bearer secret" for h in headers)

    def test_no_api_key_sends_no_authorization(self, server):
        _, headers = server({"/models": MODELS_PAYLOAD})

        discover("http://127.0.0.1:8080/v1")

        assert all("Authorization" not in h for h in headers)

    def test_a_server_without_props_still_lists_models(self, server):
        server({"/models": MODELS_PAYLOAD})

        info = discover("http://127.0.0.1:8080/v1")

        assert len(info.models) == 1
        assert info.context_tokens is None

    def test_without_props_the_training_context_is_the_fallback(self, server):
        server({"/models": MODELS_PAYLOAD})

        assert discover("http://127.0.0.1:8080/v1").effective_context == 32768

    def test_unparseable_props_is_not_fatal(self, server):
        server({"/models": MODELS_PAYLOAD, "/props": "not json at all"})

        info = discover("http://127.0.0.1:8080/v1")

        assert len(info.models) == 1
        assert info.context_tokens is None

    def test_an_unreachable_endpoint_raises(self, server):
        server({"/models": urllib.error.URLError("connection refused")})

        with pytest.raises(EndpointError, match="connection refused"):
            discover("http://127.0.0.1:8080/v1")

    def test_a_server_error_raises(self, server):
        server({})

        with pytest.raises(EndpointError):
            discover("http://127.0.0.1:8080/v1")

    def test_models_without_metadata_are_still_listed(self, server):
        server({"/models": {"data": [{"id": "gpt-4o-mini"}]}})

        info = discover("http://127.0.0.1:8080/v1")

        assert info.models[0].id == "gpt-4o-mini"
        assert info.models[0].train_context is None
        assert info.effective_context is None

    def test_junk_entries_are_skipped(self, server):
        server({"/models": {"data": ["nonsense", {"no_id": 1}, {"id": "ok"}]}})

        assert [m.id for m in discover("http://127.0.0.1:8080/v1").models] == ["ok"]

    def test_a_payload_without_data_yields_no_models(self, server):
        server({"/models": {"object": "list"}})

        assert discover("http://127.0.0.1:8080/v1").models == []

    def test_zero_values_count_as_unknown(self, server):
        server({"/models": {"data": [{"id": "m", "meta": {"n_ctx_train": 0}}]}})

        assert discover("http://127.0.0.1:8080/v1").models[0].train_context is None

    def test_a_base_url_without_v1_is_handled(self, server):
        requested, _ = server({"/models": MODELS_PAYLOAD, "/props": PROPS_PAYLOAD})

        discover("http://127.0.0.1:8080/")

        assert "http://127.0.0.1:8080/models" in requested
        assert "http://127.0.0.1:8080/props" in requested


class TestModelLabels:
    def test_billions_are_abbreviated(self):
        assert EndpointModel("m", parameters=7_615_616_512).parameters_label == "7.6B"

    def test_millions_are_abbreviated(self):
        assert EndpointModel("m", parameters=350_000_000).parameters_label == "350M"

    def test_unknown_parameters_show_a_placeholder(self):
        assert EndpointModel("m").parameters_label == "?"


class TestSuggestLimits:
    @pytest.mark.parametrize(
        "context,output,functions",
        [
            (8192, 1024, 3),
            (16384, 2048, 7),
            (32768, 4096, 15),
            (131072, 8192, 31),
        ],
    )
    def test_output_budget_and_batch_size(self, context, output, functions):
        limits = suggest_limits(context)
        assert limits.max_output_tokens == output
        assert limits.max_chunk_functions == functions

    def test_the_prompt_budget_leaves_room_for_the_answer(self):
        limits = suggest_limits(16384)
        prompt_tokens = limits.max_prompt_chars / 3
        assert prompt_tokens + limits.max_output_tokens + 600 <= 16384

    def test_a_tiny_window_still_produces_usable_limits(self):
        limits = suggest_limits(2048)
        assert limits.max_prompt_chars >= 2000
        assert limits.max_chunk_functions >= 1
        assert limits.max_output_tokens >= 256

    def test_a_huge_window_caps_the_output_budget(self):
        assert suggest_limits(1_000_000).max_output_tokens == 8192


class TestEndpointInfo:
    def test_no_models_and_no_props_means_no_context(self):
        assert EndpointInfo(base_url="http://x/v1").effective_context is None
