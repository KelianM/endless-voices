"""Offline checks for verified reuse and failure-safe fetches."""

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_sources", Path(__file__).parents[1] / "scripts" / "fetch_sources.py"
)
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    fetcher.git(root, "init", "--quiet")
    (root / "data").mkdir()
    (root / "data/example.txt").write_text('mission "Example"\n')
    for name in ("license.txt", "copyright", "credits.txt"):
        (root / name).write_text("Fixture attribution\n")
    fetcher.git(root, "add", ".")
    fetcher.git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Fixture")
    manifest = {
        "revision": fetcher.git(root, "rev-parse", "HEAD"),
        "files": [{"path": "data/example.txt", "sha256": hashlib.sha256(
            (root / "data/example.txt").read_bytes()).hexdigest()}],
    }
    return root, manifest


def test_verified_reuse_never_fetches(checkout, monkeypatch):
    root, manifest = checkout
    original = fetcher.git

    def offline_git(path, *args):
        assert "fetch" not in args
        return original(path, *args)

    monkeypatch.setattr(fetcher, "git", offline_git)
    fetcher.fetch(root, manifest)


@pytest.mark.parametrize("change", ["edit", "delete", "extra", "revision", "license"])
def test_invalid_existing_checkout_is_preserved(checkout, change):
    root, manifest = checkout
    if change == "edit":
        (root / "data/example.txt").write_text("My local work")
    elif change == "delete":
        (root / "data/example.txt").unlink()
    elif change == "extra":
        (root / "data/extra.txt").write_text("extra")
    elif change == "revision":
        manifest["revision"] = "0" * 40
    else:
        (root / "license.txt").unlink()
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        fetcher.fetch(root, manifest)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_failed_download_does_not_publish_destination(tmp_path, monkeypatch):
    def fail_git(root, *args):
        raise subprocess.CalledProcessError(1, ["git", *args])

    monkeypatch.setattr(fetcher, "git", fail_git)
    destination = tmp_path / "source"
    with pytest.raises(subprocess.CalledProcessError):
        fetcher.fetch(destination, {"revision": "a" * 40, "files": []})
    assert list(tmp_path.iterdir()) == []
