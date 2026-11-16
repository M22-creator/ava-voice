import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_hardware", Path(__file__).parent.parent / "scripts/check_hardware.py"
)
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)


def test_assess():
    ok, why = ch.assess(0.5, 50)
    assert ok and "headroom" in why
    ok, why = ch.assess(0.9, 60)
    assert ok and "little headroom" in why
    ok, _ = ch.assess(0.6, 95)
    assert ok
    ok, why = ch.assess(1.2, 100)
    assert not ok and "Too slow" in why and "96 ms" in why


def test_percentile():
    assert ch.percentile(list(range(101)), 95) == 95
    assert ch.percentile([], 95) != ch.percentile([], 95)  # nan


def test_main_without_torch_or_gpu_exits_2(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ch.main([]) == 2
    assert "PyTorch is not installed" in capsys.readouterr().out
