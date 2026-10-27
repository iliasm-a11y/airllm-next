"""CLI and campaign option contracts with no model/device initialization."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "air_llm"))
from airllm.nextgen import gguf_cli


@pytest.mark.parametrize("flags", [[], ["--speculative-depth", "2"],
    ["--kernel", "metal", "--metal-batch-backend", "tensorops", "--metal-token-tile", "8"],
    ["--kernel", "metal", "--expert-batch", "--packed-direct-io", "--grouped-direct-io"]])
def test_cli_new_controls_are_opt_in_and_forwarded(tmp_path, monkeypatch, flags):
    captured = []
    monkeypatch.setattr(gguf_cli, "_run", lambda args: captured.append(args))
    assert gguf_cli.main(["gguf-run", str(tmp_path / "checkpoint.gguf"),
        "--report", str(tmp_path / "report.json"), "--worker", *flags]) == 0
    args = captured[0]
    assert args.speculative_depth == (2 if "--speculative-depth" in flags else 0)
    assert args.metal_batch_backend == ("tensorops" if "--metal-batch-backend" in flags else "off")
    assert args.metal_token_tile == (8 if "--metal-token-tile" in flags else 4)
    assert args.grouped_direct_io == ("--grouped-direct-io" in flags)


@pytest.mark.parametrize("flags", [["--metal-batch-backend", "reuse"],
    ["--grouped-direct-io"], ["--kernel", "metal", "--grouped-direct-io", "--expert-batch"],
    ["--speculative-depth", "3"], ["--metal-token-tile", "3"]])
def test_cli_refuses_incompatible_options_before_loading(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(gguf_cli, "_run", lambda args: pytest.fail("Worker initialized"))
    with pytest.raises(SystemExit) as error:
        gguf_cli.main(["gguf-run", str(tmp_path / "checkpoint.gguf"),
            "--report", str(tmp_path / "report.json"), "--worker", *flags])
    assert error.value.code == 2 and not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("depth", [0, 2, 4, 8, 16])
def test_worker_calls_only_requested_generator_with_explicit_depth(tmp_path, monkeypatch, depth):
    from airllm.nextgen import gguf, gguf_qwen, gguf_tokenizer
    initialized, generated = [], []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.metadata = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    class Tokenizer:
        def __init__(self, metadata):
            pass
        def apply_chat_template(self, *args):
            return [1, 2, 3]
        def decode(self, values, **kwargs):
            assert values == [4, 5]
            return "verified test text"

    class Model:
        def __init__(self, path, options, **kwargs):
            initialized.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def _generate(self, mode, ids, **kwargs):
            generated.append((mode, ids, kwargs))
            return np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        def generate(self, ids, **kwargs):
            return self._generate("ordinary", ids, **kwargs)
        def generate_speculative(self, ids, **kwargs):
            return self._generate("speculative", ids, **kwargs)
        def metrics(self):
            return {"test_double": True}

    monkeypatch.setattr(gguf, "GGUFReader", Reader)
    monkeypatch.setattr(gguf_tokenizer, "GGUFTokenizer", Tokenizer)
    monkeypatch.setattr(gguf_qwen, "NativeGGUFQwenModel", Model)
    report = tmp_path / "report.json"
    assert gguf_cli.main(["gguf-run", str(tmp_path / "checkpoint.gguf"), "--worker",
        "--report", str(report), "--speculative-depth", str(depth)]) == 0
    mode, ids, options = generated[0]
    assert mode == ("speculative" if depth else "ordinary") and ids == [[1, 2, 3]]
    assert options == dict(max_new_tokens=32, eos_token_id=[248046, 248044],
        **(dict(max_depth=depth, candidate_depths=(depth,)) if depth else {}))
    assert initialized[0]["metal_batch_backend"] == "off"
    assert initialized[0]["grouped_direct_io"] is False
    assert json.loads(report.read_text())["generated_ids"] == [4, 5]


def test_campaign_accepts_only_explicit_supported_h3_controls(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    campaign = importlib.import_module("campaign_research51")
    arguments = ["--metal-batch-backend", "tensorops", "--metal-token-tile", "16",
        "--speculative-depth", "8", "--grouped-direct-io", "--expert-batch", "--packed-direct-io"]
    assert campaign.validate_job_args(arguments) == arguments
    for invalid in (["--speculative-depth", "1"], ["--metal-batch-backend", "auto"],
                    ["--metal-token-tile", "32"], ["--metal-batch", "tensorops"]):
        with pytest.raises(ValueError):
            campaign.validate_job_args(invalid)


def test_observer_reset_separates_rejected_paths_without_erasing_costs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    module = importlib.import_module("hyperstream2_observer")
    observer = module.RuntimeObserver(SimpleNamespace())
    observer._routing = {"0": {"2": [1, 3]}}
    observer._totals["layer.moe"]["calls"] = 7
    observer.reset_routes()
    assert observer.routing == {}
    assert observer.summary()["methods"]["layer.moe"]["calls"] == 7
