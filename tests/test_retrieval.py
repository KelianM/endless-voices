"""Protect scene eligibility across vector ranking, tool calls and saved indexes."""

from copy import deepcopy

import pytest

from endless_voices.context import ContextPool
from endless_voices.retrieval import RetrievalSession, SceneDatabase, passage_chunks, scene_id


class Encoder:
    identity = "test-semantic-space-v1"

    def __init__(self):
        self.inputs = []

    def embed(self, texts, *, kind):
        self.inputs.extend(texts)
        return [[1.0, 0.0] if "starship" in text else [0.0, 1.0] for text in texts]


def block(mission, text, line=1, **extra):
    return {"mission": mission, "heading": mission, "path": "missions.txt",
            "passages": [{"line": line, "text": text, "role": "passage"}], **extra}


def pool():
    return ContextPool(
        "sample", "conversation", "current", "Instructions\nEarlier passages:\n",
        [{"role": "user", "content": "A starship arrives."}],
        [block("parent", "A starship departs."), block("ancestor", "Old story", 2)],
        {"current": ["parent"], "parent": ["ancestor"]}, {"<planet>": "Earth"},
        [block("other", "A reference voice", 3, reference=True, conversation=3, state={})],
    )


def test_ineligible_scene_cannot_affect_ranking_or_be_read_by_guessed_id(tmp_path):
    source, encoder = pool(), Encoder()
    hidden = block("current", "The starship's hidden continuation", 40)
    incompatible = deepcopy(source.blocks[0])
    incompatible["passages"].append({"line": 41, "text": "Secret", "role": "passage"})
    with SceneDatabase.create(tmp_path / "scenes.db",
                              [hidden, incompatible, *source.blocks, *source.references],
                              encoder) as database:
        session = RetrievalSession(database, source)
        encoder.inputs.clear()
        matches = session.search_context(limit=1)
        assert [match["scene_id"] for match in matches] == [scene_id(source.blocks[0])]
        assert encoder.inputs == [source.encounter[0]["content"]]
        for sid in [scene_id(hidden), scene_id(incompatible),
                    passage_chunks(hidden)[0]["id"], "invented"]:
            with pytest.raises(ValueError, match="not eligible"):
                session.read_context([sid])
        expected = session.search_context()
        session.selected_scene_ids = {expected[0]["scene_id"]}
        assert session.search_context(limit=1)[0]["scene_id"] == expected[1]["scene_id"]
        assert all(doc["scene_id"] not in session.selected_scene_ids
                   for doc in session.search_context(expand_scenes=False))
        assert session.read_context([expected[0]["scene_id"]])


    hidden["passages"][0]["text"] = "A completely different hidden target."
    with SceneDatabase.create(tmp_path / "changed.db",
                              [hidden, *source.blocks, *source.references], encoder) as database:
        assert RetrievalSession(database, source).search_context() == expected


def test_history_depth_and_reference_tools_keep_distinct_provenance(tmp_path):
    source = pool()
    with SceneDatabase.create(tmp_path / "scenes.db", source.blocks + source.references,
                              Encoder()) as database:
        session = RetrievalSession(database, source)
        assert session.get_history(0) == []
        history = session.call_tool("get_history", {"depth": 1})
        assert [entry["mission"] for entry in history] == ["parent"]
        assert history[0]["depth"] == 1
        assert "A starship departs." in history[0]["text"]
        assert len(session.get_history(2)) == 2
        references = session.call_tool("search_context", {"kind": "reference"})
        assert len(references) == 1
        assert references[0]["kind"] == "reference"
        assert "not current playthrough" in references[0]["usage"]
        read = session.call_tool("read_context", {"ids": [references[0]["id"]]})
        assert read[0]["lines"] == [3]
        assert read[0]["path"] == "missions.txt"
        assert "A reference voice" in read[0]["text"]
        assert session.trace[-1]["result_ids"] == [references[0]["id"]]
        assert {tool["function"]["name"] for tool in session.tool_schemas} == {
            "get_history", "search_context", "read_context"}


def test_scene_index_never_substitutes_another_rendered_state_or_partial_scene(tmp_path):
    source = pool()
    other = deepcopy(source)
    other.blocks[0]["passages"][0]["text"] = "A starship departs from Mars."
    with SceneDatabase.create(tmp_path / "scenes.db", source.blocks + source.references,
                              Encoder()) as database:
        with pytest.raises(ValueError, match="absent"):
            RetrievalSession(database, other)
        session = RetrievalSession(database, source)
        source.blocks[0]["passages"][0]["text"] = "Mutation after session creation"
        source.encounter[0]["content"] = "Mutated encounter"
        assert session.initial_messages()[1]["content"] == "A starship arrives."
        assert "Mutation" not in session.get_history(1)[0]["text"]
        read = session.get_history(1)
        read[0]["lines"].append(99)
        assert session.get_history(1)[0]["lines"] == [1]


def test_saved_vectors_require_the_same_embedding_space(tmp_path):
    source, encoder = pool(), Encoder()
    path = tmp_path / "scenes.db"
    SceneDatabase.create(path, source.blocks + source.references, encoder).close()
    encoder.identity = "different-model"
    with pytest.raises(ValueError, match="model differs"):
        SceneDatabase(path, encoder)
    encoder.identity = "test-semantic-space-v1"
    with SceneDatabase(path, encoder) as database:
        assert RetrievalSession(database, source).search_context(limit=1)[0]["mission"] == "parent"
    with pytest.raises(FileExistsError):
        SceneDatabase.create(path, source.blocks, encoder)


class Counter:
    def messages(self, messages):
        return 20 + sum(len(message["content"]) for message in messages)


def test_static_selection_prioritizes_nearby_history_then_similarity_with_wrappers(tmp_path):
    source = pool()
    source.blocks[0]["passages"][0]["text"] = "Nearby history"
    source.references[0]["passages"][0]["text"] = "A starship reference"
    source.blocks.append(block("huge", "starship " * 1000, 10))
    with SceneDatabase.create(tmp_path / "scenes.db", source.blocks + source.references,
                              Encoder()) as database:
        session = RetrievalSession(database, source)
        wanted = [scene_id(source.blocks[0]), scene_id(source.references[0])]
        expected = session.messages_for(wanted)
        budget = Counter().messages(expected)
        selected = session.static_messages(Counter(), budget, depth=1)
        assert selected == expected
        assert Counter().messages(selected) == budget
        assert "independent writing references" in selected[0]["content"]
        assert selected[1:] == source.encounter
        assert session.trace[-1]["result_ids"] == wanted
        assert len(source.blocks) == 3


def test_static_budget_rejects_oversized_encounter_without_truncation(tmp_path):
    source = pool()
    with SceneDatabase.create(tmp_path / "scenes.db", source.blocks, Encoder()) as database:
        source.references = []
        session = RetrievalSession(database, source)
        with pytest.raises(ValueError, match="visible encounter exceed"):
            session.static_messages(Counter(), Counter().messages(session.initial_messages()) - 1)
        assert session.initial_messages()[1:] == source.encounter


def test_static_selection_does_not_inherit_tool_result_limit(tmp_path):
    source = pool()
    source.blocks = [block("parent", f"starship {i}", i) for i in range(101)]
    source.references = []
    with SceneDatabase.create(tmp_path / "scenes.db", source.blocks, Encoder()) as database:
        session = RetrievalSession(database, source)
        assert session.static_messages(Counter()) == session.messages_for(
            [scene_id(block) for block in source.blocks])
        assert len(session.trace[-1]["result_ids"]) == 101


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_embeddings_fail_before_creating_an_index(tmp_path, vector):
    class InvalidEncoder(Encoder):
        def embed(self, texts, *, kind):
            return [vector for text in texts]

    path = tmp_path / "scenes.db"
    with pytest.raises(ValueError, match="finite"):
        SceneDatabase.create(path, pool().blocks, InvalidEncoder())
    assert not path.exists()


def test_hidden_target_in_an_omitted_reference_prevents_publication(tmp_path):
    from test_dataset_builder import config, corpus, spec

    from endless_voices.dataset.builder import DatasetBuilder

    source = corpus(tmp_path, "mission Current\n\ton offer\n\t\tconversation\n\t\t\t`Target.`\n")
    builder = DatasetBuilder(source, config({"Current": spec()}))
    leaked = block("other", "Hidden target " * 10000, 4,
                   reference=True, conversation=3, state={})
    builder.sampler.references = lambda *args: [leaked]
    with pytest.raises(ValueError, match="Target source paragraph appears in eligible context"):
        builder.build(tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def save_local_encoder(model_path, *, sharded=False):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    model_path.mkdir()
    vocab = model_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nstarship\narrives\nold\nstory\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(model_path)
    model = BertModel(BertConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=16))
    model.save_pretrained(model_path, max_shard_size="2KB" if sharded else "5GB")
    return model_path


def test_local_transformer_encoder_loads_and_masks_padding_without_downloads(tmp_path):
    import torch

    from endless_voices.retrieval import TransformersEmbedder

    model_path = save_local_encoder(tmp_path / "encoder")
    encoder = TransformersEmbedder.from_pretrained(model_path, max_length=512, device="cpu")
    encoder.tokenizer.padding_side = "right"
    alone = encoder.embed(["starship"], kind="document")[0]
    padded = encoder.embed(["starship", "a starship arrives old story"], kind="document")[0]
    assert len(alone) == 8
    assert torch.allclose(torch.tensor(alone), torch.tensor(padded), atol=1e-6)
    reloaded = TransformersEmbedder.from_pretrained(model_path, max_length=512, device="cpu")
    assert encoder.identity == reloaded.identity
    source = pool()
    with SceneDatabase.create(tmp_path / "scenes.db", source.blocks + source.references,
                              encoder) as database:
        matches = RetrievalSession(database, source).search_context(limit=2)
        assert len(matches) == 2
        assert all(-1.00001 <= match["score"] <= 1.00001 for match in matches)


def test_search_finds_late_passages_and_expands_only_the_eligible_parent(tmp_path):
    source = pool()
    source.blocks = [block("parent", "An opening without a match", 1)]
    source.blocks[0]["passages"].extend(
        {"line": line, "text": "An unrelated passage " * 100, "role": "passage"}
        for line in range(2, 12))
    source.blocks[0]["passages"].append(
        {"line": 12, "text": "The starship waits at the final port.", "role": "passage"})
    with SceneDatabase.create(tmp_path / "chunks.db", source.blocks + source.references,
                              Encoder()) as database:
        session = RetrievalSession(database, source)
        hit = session.search_context("starship", kind="history", limit=1, expand_scenes=False)[0]
        assert hit["lines"] == [12]
        assert hit["passage_index"] == 11
        assert hit["scene_id"] == scene_id(source.blocks[0])
        assert session.read_context([hit["id"]])[0]["text"] == hit["excerpt"]
        parent = session.read_context([hit["scene_id"]])[0]
        assert parent["lines"] == list(range(1, 13))
        assert "opening without" in parent["text"]
        assert "final port" in parent["text"]


@pytest.mark.parametrize("mask,last_values", [
    ([[1, 1, 0], [1, 1, 1]], [2.0, 3.0]),
    ([[0, 1, 1], [1, 1, 1]], [3.0, 3.0]),
])
def test_qwen_pooling_uses_last_nonpadding_token_and_instructs_only_queries(mask, last_values):
    from types import SimpleNamespace

    import torch

    from endless_voices.retrieval import TransformersEmbedder

    calls = []

    def tokenizer(texts, **kwargs):
        calls.append((texts, kwargs))
        return {"attention_mask": torch.tensor(mask)}

    class Model:
        config = SimpleNamespace(max_position_embeddings=32)
        dtype = torch.float32
        device = torch.device("cpu")

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor([
                [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
                [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
            ]))

    encoder = TransformersEmbedder(tokenizer, Model(), "fixture", max_length=32,
                                   query_instruction="Find relevant authored passages.")
    with pytest.raises(TypeError, match="kind"):
        encoder.embed(["missing input kind"])
    with pytest.raises(ValueError, match="kind must be document or query"):
        encoder.embed(["invalid input kind"], kind="passage")
    actual = encoder.embed(["first document", "second document"], kind="document")
    expected = torch.nn.functional.normalize(torch.tensor([[v, 1.0] for v in last_values]), dim=1)
    assert torch.allclose(torch.tensor(actual), expected)
    assert calls[-1][0] == ["first document", "second document"]
    encoder.embed(["first query", "second query"], kind="query")
    assert calls[-1][0] == [
        "Instruct: Find relevant authored passages.\nQuery:first query",
        "Instruct: Find relevant authored passages.\nQuery:second query"]
    assert all(call[1]["truncation"] is False for call in calls)
    encoder.max_length = 2
    with pytest.raises(ValueError, match="Document contains 3 tokens.*no text was truncated"):
        encoder.embed(["first document", "second document"], kind="document")
    with pytest.raises(ValueError, match="Instructed query contains 3 tokens.*no text"):
        encoder.embed(["first query", "second query"], kind="query")


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_incomplete_or_changed_passages_cannot_silently_change_retrieval(tmp_path, damage):
    import json
    import sqlite3

    source = pool()
    source.blocks[0]["passages"][0]["role"] = "option"
    chunk = passage_chunks(source.blocks[0])[0]
    assert chunk["text"] == "parent\nPlayer choice: A starship departs."
    path = tmp_path / "passages.db"
    SceneDatabase.create(path, source.blocks + source.references, Encoder()).close()
    with sqlite3.connect(path) as connection:
        if damage == "missing":
            connection.execute("DELETE FROM passages WHERE id = ?", (chunk["id"],))
        else:
            changed = {**chunk, "text": "Different authored text"}
            connection.execute("UPDATE passages SET chunk = ? WHERE id = ?",
                               (json.dumps(changed), chunk["id"]))
    with SceneDatabase(path, Encoder()) as database:
        with pytest.raises(ValueError, match="missing an eligible passage|differs from its parent"):
            RetrievalSession(database, source)


def test_scene_expansion_returns_neighboring_answer_and_counts_distinct_scenes(tmp_path):
    class RankedEncoder(Encoder):
        def embed(self, texts, *, kind):
            return [[0.8, 0.6] if "other vessel" in text else
                    [1.0, 0.0] if "starship" in text else [0.0, 1.0] for text in texts]

    source = pool()
    first = block("parent", "Which starship carries the ambassador?", 10)
    first["passages"].extend([
        {"line": 11, "text": "The starship is departing soon.", "role": "passage"},
        {"line": 12, "text": "The ambassador is aboard the Peregrine.", "role": "passage"},
    ])
    second = block("ancestor", "The other vessel is still in port.", 20)
    source.blocks = [first, second]
    hidden_variant = deepcopy(first)
    hidden_variant["passages"].append(
        {"line": 13, "text": "A hidden future betrayal.", "role": "passage"})
    with SceneDatabase.create(tmp_path / "scenes.db",
                              [*source.blocks, *source.references, hidden_variant],
                              RankedEncoder()) as database:
        session = RetrievalSession(database, source)
        expanded = session.call_tool("search_context", {
            "query": "starship", "kind": "history", "limit": 2})
        assert [document["id"] for document in expanded] == [scene_id(first), scene_id(second)]
        assert expanded[0]["lines"] == [10, 11, 12]
        assert "aboard the Peregrine" in expanded[0]["text"]
        assert "hidden future betrayal" not in expanded[0]["text"]
        match = expanded[0]["matched_passage"]
        assert match["line"] in {10, 11}
        assert match["score"] == expanded[0]["score"] == 1.0
        trace = deepcopy(session.trace[-1])
        assert trace["arguments"]["expand_scenes"] is True
        assert trace["matched_passages"][0] == {"scene_id": scene_id(first), **match}
        assert trace["result_ids"] == [scene_id(first), scene_id(second)]
        raw = session.search_context("starship", kind="history", limit=2, expand_scenes=False)
        assert len(raw) == 2
        assert all(document["scene_id"] == scene_id(first) for document in raw)
        assert raw[0]["id"] == match["id"]
        assert all("matched_passage" not in document and "text" not in document for document in raw)
        assert all("aboard the Peregrine" not in document["excerpt"] for document in raw)
        assert session.trace[-1]["arguments"]["expand_scenes"] is False
        expanded[0]["matched_passage"]["line"] = 999
        assert trace == session.trace[-2]
        with pytest.raises(ValueError, match="not eligible"):
            session.read_context([scene_id(hidden_variant)])
        with pytest.raises(ValueError, match="expand_scenes must be a boolean"):
            session.search_context(expand_scenes="false")


@pytest.mark.parametrize("sharded", [False, True])
def test_encoder_identity_ignores_unloaded_files_and_tracks_dependencies(tmp_path, sharded):
    import json

    from safetensors.torch import load_file, save_file

    from endless_voices.retrieval import TransformersEmbedder

    model_path = save_local_encoder(tmp_path / "encoder", sharded=sharded)

    def load():
        return TransformersEmbedder.from_pretrained(model_path, max_length=512, device="cpu")

    original = load()
    path = tmp_path / "scenes.db"
    source = pool()
    SceneDatabase.create(path, source.blocks + source.references, original).close()
    (model_path / "README.md").write_text("Local model notes")
    (model_path / "run.log").write_text("An inference log")
    (model_path / ".cache").mkdir()
    (model_path / ".cache" / "download.metadata").write_text("Updated timestamp")
    (model_path / "pytorch_model.bin").write_bytes(b"Unselected weight format")
    (model_path / "generation_config.json").write_text('{"max_length": 10}')
    (model_path / "unreferenced.safetensors").write_bytes(b"Unselected weights")
    unchanged = load()
    assert unchanged.identity == original.identity
    with SceneDatabase(path, unchanged) as database:
        assert RetrievalSession(database, source).search_context(limit=1)

    if sharded:
        index = json.loads((model_path / "model.safetensors.index.json").read_text())
        weight_path = model_path / next(iter(index["weight_map"].values()))
    else:
        weight_path = model_path / "model.safetensors"
    for dependency in (model_path / "config.json", model_path / "tokenizer.json", weight_path):
        previous = dependency.read_bytes()
        if dependency.name == "config.json":
            content = json.loads(previous)
            content["hidden_act"] = "relu"
            dependency.write_text(json.dumps(content))
        elif dependency.name == "tokenizer.json":
            content = json.loads(previous)
            content["model"]["vocab"]["starship"], content["model"]["vocab"]["arrives"] = (
                content["model"]["vocab"]["arrives"], content["model"]["vocab"]["starship"])
            dependency.write_text(json.dumps(content))
        else:
            weights = load_file(dependency)
            first = next(iter(weights))
            weights[first] = weights[first] + 0.25
            save_file(weights, dependency, metadata={"format": "pt"})
        changed = load()
        assert changed.identity != original.identity
        with pytest.raises(ValueError, match="model differs"):
            SceneDatabase(path, changed)
        dependency.write_bytes(previous)
        assert load().identity == original.identity


@pytest.mark.parametrize("destination", ["file", "directory", "dangling_symlink"])
def test_existing_index_destination_fails_before_embedding(tmp_path, destination):
    path = tmp_path / "existing"
    if destination == "file":
        path.write_bytes(b"Existing index contents")
    elif destination == "directory":
        path.mkdir()
    else:
        path.symlink_to(tmp_path / "missing-target")
    encoder = Encoder()
    with pytest.raises(FileExistsError):
        SceneDatabase.create(path, pool().blocks, encoder)
    assert encoder.inputs == []
    if destination == "file":
        assert path.read_bytes() == b"Existing index contents"
    elif destination == "directory":
        assert path.is_dir()
    else:
        assert path.is_symlink()


def test_index_created_during_embedding_is_not_overwritten(tmp_path):
    path = tmp_path / "scenes.db"

    class RacingEncoder(Encoder):
        def embed(self, texts, *, kind):
            path.write_bytes(b"Another writer's index")
            return super().embed(texts, kind=kind)

    with pytest.raises(FileExistsError):
        SceneDatabase.create(path, pool().blocks, RacingEncoder())
    assert path.read_bytes() == b"Another writer's index"


def test_static_vectors_serve_distinct_encounters_without_leaking_sampled_values(tmp_path):
    from endless_voices.retrieval import authored_scene

    template = "Captain <first> visits <origin> aboard <npc>."

    def encounter(name, origin):
        source = pool()
        source.blocks = [block("parent", f"Captain {name} visits {origin} aboard Finch.",
                               substitutions={"<first>": name, "<origin>": origin,
                                              "<npc>": "Finch"})]
        source.blocks[0]["passages"][0]["role"] = "option"
        source.blocks[0]["source_passages"] = [
            {"line": 1, "text": template, "role": "option"}]
        source.references = []
        return source

    first, second = encounter("Alex", "Earth"), encounter("Sam", "Mars")
    encoder = Encoder()
    with SceneDatabase.create(tmp_path / "static.db", first.blocks + second.blocks,
                              encoder) as database:
        assert encoder.inputs == ["parent\nPlayer choice: " + template]
        stored = database.connection.execute("SELECT block FROM scenes").fetchall()
        assert len(stored) == 1
        assert all(value not in stored[0][0] for value in ("Alex", "Sam", "Earth", "Mars",
                                                          "Finch", "substitutions"))
        for source in (first, second):
            session = RetrievalSession(database, source)
            hit = session.search_context("Captain", expand_scenes=False)[0]
            expected = "parent\nPlayer choice: " + source.blocks[0]["passages"][0]["text"]
            assert hit["excerpt"] == expected
            assert session.read_context([hit["id"]])[0]["text"] == expected
            assert session.get_history(1)[0]["text"] == expected
            assert expected in session.messages_for([hit["scene_id"]])[0]["content"]
        combined = deepcopy(first)
        combined.blocks.extend(second.blocks)
        session = RetrievalSession(database, combined)
        assert len(session.get_history(1)) == 2
        assert {doc["text"] for doc in session.get_history(1)} == {
            "parent\nPlayer choice: " + source.blocks[0]["passages"][0]["text"]
            for source in (first, second)}
        changed = deepcopy(first)
        changed.blocks[0]["source_passages"][0]["text"] += " New authored sentence."
        with pytest.raises(ValueError, match="absent"):
            RetrievalSession(database, changed)
        assert authored_scene(first.blocks[0]) == authored_scene(second.blocks[0])


def test_index_rejects_rendered_context_without_authored_passages(tmp_path):
    rendered = block("parent", "Captain Alex arrives.", substitutions={"<first>": "Alex"})
    encoder = Encoder()
    with pytest.raises(ValueError, match="missing authored passages"):
        SceneDatabase.create(tmp_path / "missing.db", [rendered], encoder)
    assert not encoder.inputs
