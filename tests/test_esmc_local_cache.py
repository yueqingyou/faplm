from __future__ import annotations

from pathlib import Path

import faesm.esmc as esmc_module


def test_load_esm_tokenizer_uses_local_files_only(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class _DummyTokenizer:
        pass

    def _fake_from_pretrained(model_id: str, *, local_files_only: bool):
        calls.append((model_id, local_files_only))
        return _DummyTokenizer()

    monkeypatch.setattr(esmc_module.AutoTokenizer, "from_pretrained", _fake_from_pretrained)

    tokenizer = esmc_module._load_esm_tokenizer()

    assert isinstance(tokenizer, _DummyTokenizer)
    assert calls == [("facebook/esm2_t33_650M_UR50D", True)]


def test_data_root_uses_local_snapshot_download(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_snapshot_download_local(repo_id: str) -> str:
        calls.append(repo_id)
        return str(tmp_path)

    monkeypatch.setattr(esmc_module, "_snapshot_download_local", _fake_snapshot_download_local)
    esmc_module.data_root.cache_clear()

    path = esmc_module.data_root("esmc-600")

    assert path == tmp_path
    assert calls == ["EvolutionaryScale/esmc-600m-2024-12"]


def test_esmc_600m_loads_weights_with_weights_only(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _DummyModel:
        def eval(self):
            return self

        def load_state_dict(self, state_dict) -> None:
            captured["state_dict"] = state_dict

    def _fake_esmc(**_kwargs):
        return _DummyModel()

    def _fake_torch_load(path, *, map_location, weights_only):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return {"ok": True}

    monkeypatch.setattr(esmc_module, "ESMC", _fake_esmc)
    monkeypatch.setattr(esmc_module, "data_root", lambda _model: tmp_path)
    monkeypatch.setattr(esmc_module.torch, "load", _fake_torch_load)

    model = esmc_module.ESMC_600M_202412(device="cpu", use_flash_attn=False)

    assert isinstance(model, _DummyModel)
    assert captured["path"] == tmp_path / "data/weights/esmc_600m_2024_12_v0.pth"
    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is True
    assert captured["state_dict"] == {"ok": True}
