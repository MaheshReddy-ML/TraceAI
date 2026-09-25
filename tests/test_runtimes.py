import json
from pathlib import Path

import pytest

from traceai.errors import RuntimeUnavailableError, UnsupportedCapabilityError
from traceai.probes import BehaviorProbe, ProbeCase, get_probe
from traceai.runtimes import create_runtime, discover_models
from traceai.runtimes.mock import MockRuntime
from traceai.schemas import Capability, GenerationRequest, ModelSpec


def test_mock_runtime_is_repeatable_and_only_exposes_outputs():
    runtime = create_runtime(ModelSpec(runtime="mock", model="fixture"))
    runtime.load()
    request = GenerationRequest(
        system="system",
        prompt="prompt",
        seed=42,
        max_new_tokens=16,
        temperature=0,
        probe="baseline",
        case_id="baseline-1",
        checkpoint="1",
    )
    assert runtime.generate(request).text == runtime.generate(request).text
    assert runtime.capabilities() == {Capability.OUTPUTS}
    with pytest.raises(UnsupportedCapabilityError):
        runtime.inspect(Capability.HIDDEN_STATES)


def test_ollama_rejects_nonlocal_endpoint_before_network():
    with pytest.raises(RuntimeUnavailableError, match="local HTTP"):
        create_runtime(ModelSpec(runtime="ollama", model="model", endpoint="https://example.com"))


def test_discovery_lists_only_cached_text_generation_models(tmp_path: Path, monkeypatch):
    root = tmp_path / "hub"
    for name, architecture in (
        ("models--org--text", "ExampleForCausalLM"),
        ("models--org--vision", "ExampleForConditionalGeneration"),
    ):
        snapshot = root / name / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps({"architectures": [architecture]}))
        (snapshot / "model.safetensors").write_bytes(b"fixture")
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setenv("OLLAMA_HOST", "https://example.com")
    found = discover_models()
    assert [model["model"] for model in found] == ["org/text"]


def test_discovery_lists_configured_local_gguf_files(tmp_path: Path, monkeypatch):
    root = tmp_path / "gguf"
    root.mkdir()
    (root / "fixture.gguf").write_bytes(b"test")
    monkeypatch.setenv("TRACEAI_MODEL_DIR", str(root))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    monkeypatch.setenv("OLLAMA_HOST", "https://example.com")
    assert discover_models() == [
        {
            "runtime": "llama_cpp",
            "model": "fixture",
            "path": str((root / "fixture.gguf").resolve()),
            "size_bytes": 4,
        }
    ]


def test_selected_plugins_load_without_core_registry_edits(monkeypatch):
    class CustomProbe(BehaviorProbe):
        name = "custom"

        def cases(self):
            return [ProbeCase("custom-1", "", "ping")]

        def evaluate(self, case, output):
            return 0.0, "fixture"

    class CustomRuntime(MockRuntime):
        pass

    class EntryPoint:
        def __init__(self, name, implementation):
            self.name = name
            self.implementation = implementation

        def load(self):
            return self.implementation

    monkeypatch.setattr(
        "traceai.probes.entry_points", lambda group: [EntryPoint("custom", CustomProbe)]
    )
    monkeypatch.setattr(
        "traceai.runtimes.entry_points", lambda group: [EntryPoint("custom", CustomRuntime)]
    )
    assert get_probe("custom").cases()[0].prompt == "ping"
    assert isinstance(create_runtime(ModelSpec(runtime="custom", model="fixture")), CustomRuntime)
