"""Protect split selection, literal sample text and generation-run matching in the reader."""

import json
from pathlib import Path

import pytest

from endless_voices import view_samples as viewer

FIXTURE = Path(__file__).parent / "fixtures/contracts/manifest.json"


def payload(output):
    text = output.read_text()
    return json.loads(
        text.split('<script id="samples-data" type="application/json">')[1].split("</script>")[0]
    )


def test_default_export_excludes_held_out_content_and_preserves_history(tmp_path):
    output = tmp_path / "samples.html"
    assert viewer.build_viewer(FIXTURE, output) == 1
    data = payload(output)
    original = json.loads((FIXTURE.parent / "train.jsonl").read_text())
    original.pop("evaluation")
    original["player_choices"] = None
    assert data["samples"] == [original]
    assert data["runs"] == []
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        viewer.build_viewer(FIXTURE, output)
    assert output.read_bytes() == before


def test_literal_script_delimiters_cannot_escape_sample_data(tmp_path):
    import shutil

    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURE.parent, dataset)
    manifest = dataset / "manifest.json"
    source = dataset / "train.jsonl"
    record = json.loads(source.read_text())
    hostile = '</script><script>alert("sample text")</script>'
    record["messages"][-1]["content"] = hostile
    source.write_text(json.dumps(record) + "\n")
    declaration = json.loads(manifest.read_text())
    declaration["artifacts"]["train.jsonl"] = viewer.sha256(source)
    manifest.write_text(json.dumps(declaration))
    output = tmp_path / "samples.html"
    viewer.build_viewer(manifest, output)
    assert hostile not in output.read_text()
    assert payload(output)["samples"][0]["messages"][-1]["content"] == hostile


def test_run_join_keeps_failures_and_missing_responses_visible(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "interrupted",
                "dataset": {
                    "manifest_sha256": viewer.sha256(FIXTURE),
                    "sample_ids": ["fixture-train", "fixture-validation"],
                },
            }
        )
    )
    (run / "responses.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "fixture-train",
                "status": "failed",
                "response": "partial text",
                "error": {"stage": "generation", "type": "OutputLimit", "message": "capped"},
            }
        )
        + "\n"
    )
    output = tmp_path / "samples.html"
    assert viewer.build_viewer(FIXTURE, output, "all", [run]) == 3
    data = payload(output)
    assert data["runs"][0]["responses"]["fixture-train"]["error"]["type"] == "OutputLimit"
    assert "fixture-validation" in data["runs"][0]["selected_ids"]
    assert "fixture-validation" not in data["runs"][0]["responses"]
    assert "fixture-test" not in data["runs"][0]["selected_ids"]


@pytest.mark.parametrize("problem", ["manifest", "duplicate", "hash"])
def test_unrelated_or_modified_responses_are_rejected(tmp_path, problem):
    run = tmp_path / "run"
    run.mkdir()
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "dataset": {
            "manifest_sha256": viewer.sha256(FIXTURE),
            "sample_ids": ["fixture-train"],
        },
    }
    row = json.dumps({"sample_id": "fixture-train", "status": "ok", "response": "Hello"})
    (run / "responses.jsonl").write_text(row + "\n")
    if problem == "manifest":
        metadata["dataset"]["manifest_sha256"] = "wrong"
    elif problem == "duplicate":
        (run / "responses.jsonl").write_text(row + "\n" + row + "\n")
    else:
        metadata["artifacts_sha256"] = {"responses.jsonl": "wrong"}
    (run / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        viewer.build_viewer(FIXTURE, tmp_path / "samples.html", run_directories=[run])


@pytest.mark.parametrize("model", ["gpt-6-luna", "claude-sonnet-5-5"])
def test_live_turns_keep_evidence_without_sending_authored_target(tmp_path, monkeypatch, model):
    # A follow-up must retain tool evidence while the withheld answer stays server-side.
    from types import SimpleNamespace

    from endless_voices import live_viewer, providers

    data, datasets = viewer.viewer_data(FIXTURE)
    data["live"] = {"models": [{"id": model}]}
    data["samples"][0]["player_choices"] = {
        "prefix": "Encounter", "options": ["Original choice", "Another choice"],
        "selected": "Original choice"}
    session = SimpleNamespace(
        history_count=1, tool_schemas=[], selected_scene_ids=set(),
        initial_messages=lambda: [{"role": "system", "content": "Scene instructions"},
                                  {"role": "user", "content": "Encounter\n\nOriginal choice"}])
    def retrieve(name, arguments):
        if arguments.get("followup"):
            assert session.selected_scene_ids == {"evidence"}
        return [{"id": "evidence", "scene_id": "evidence", "text": "Earlier event"}]
    session.call_tool = retrieve
    monkeypatch.setattr(datasets["train"], "retrieval_session", lambda i, db: session)
    app = live_viewer.LiveViewer(data, datasets, {"context": {
        "hosted_max_input_tokens": {model: 8192}}}, tmp_path, 1)
    app.database = object()
    app.counter = SimpleNamespace(messages=lambda messages: 10)
    monkeypatch.setattr(providers, "load_key", lambda *args: "test")
    bodies = []
    def ask(root, path, body, *args):
        from copy import deepcopy
        bodies.append(deepcopy(body))
        call_turn = len(bodies) % 2 == 1
        call = {"followup": len(bodies) > 2}
        if model.startswith("claude"):
            content = ([{"type": "tool_use", "id": str(len(bodies)), "name": "read_context",
                         "input": call}] if call_turn else [{"type": "text", "text": "Reply"}])
            response = {"content": content}
        else:
            content = ([{"type": "function_call", "call_id": str(len(bodies)),
                         "name": "read_context", "arguments": json.dumps(call)}] if call_turn
                       else [{"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": "Reply"}]}])
            response = {"output": content}
        return {"status": "ok", "api_response": response, "estimated_cost_usd": 0.001}
    monkeypatch.setattr(live_viewer, "hosted_request", ask)
    first = app.reply({"model": model, "sample_id": "fixture-train", "text": "Another choice"})
    assert "Original choice" not in json.dumps(bodies)
    assert "Encounter\\n\\nAnother choice" in json.dumps(bodies)
    second = app.reply({"model": model, "sample_id": "fixture-train",
                        "session_id": first["session_id"], "text": "My follow-up"})
    assert second["session_id"] == first["session_id"]
    assert "Earlier event" in json.dumps(bodies[2])
    assert "My follow-up" in json.dumps(bodies[2])
    assert datasets["train"].target(0) not in json.dumps(bodies)
    with pytest.raises(ValueError, match="does not belong"):
        app.reply({"model": model, "sample_id": "wrong", "session_id": first["session_id"]})


def test_catalogue_paginates_and_excludes_non_chat_models(monkeypatch):
    # Missing pagination hides accessible Claude models; non-chat IDs cannot run the tool loop.
    from io import StringIO

    from endless_voices import providers

    pages = iter([
        {"data": [{"id": "claude-one"}], "has_more": True, "last_id": "claude-one"},
        {"data": [{"id": "claude-two"}], "has_more": False},
        {"data": [{"id": name} for name in ["gpt-6-luna", "gpt-image-1", "tts-1"]]},
    ])
    urls = []
    def fetch(request, timeout):
        urls.append(request.full_url)
        return StringIO(json.dumps(next(pages)))
    monkeypatch.setattr(providers.urllib.request, "urlopen", fetch)
    assert providers.list_models("anthropic", "test") == ["claude-one", "claude-two"]
    assert urls[1].endswith("?after_id=claude-one")
    assert providers.list_models("openai", "test") == ["gpt-6-luna"]
