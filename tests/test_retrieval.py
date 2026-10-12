"""Protect scene eligibility across vector ranking, tool calls and saved indexes."""

from copy import deepcopy

import pytest

from endless_voices.context import ContextPool, MissionDepth
from endless_voices.retrieval import RetrievalSession, SceneDatabase, scene_id


class Encoder:
    identity = "test-semantic-space-v1"

    def __init__(self):
        self.inputs = []

    def embed(self, texts):
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
        assert [match["id"] for match in matches] == [scene_id(source.blocks[0])]
        assert encoder.inputs == [source.encounter[0]["content"]]
        for sid in [scene_id(hidden), scene_id(incompatible), "invented"]:
            with pytest.raises(ValueError, match="not eligible"):
                session.read_context([sid])
        expected = session.search_context()

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


def test_fixed_sampler_retains_all_eligible_material_for_later_retrieval():
    class Counter:
        def messages(self, messages):
            return sum(len(message["content"]) for message in messages)

        def text(self, text):
            return len(text)

    source = pool()
    budget = Counter().messages(source.messages({"parent"}))
    selection = MissionDepth(1, budget, "fixed").select(source, Counter())
    assert selection.messages == source.messages({"parent"})
    assert selection.pool == source
    assert len(selection.omitted) == 2
    assert "pool" not in selection.evidence()
    source.blocks.clear()
    assert len(selection.pool.blocks) == 2


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_embeddings_fail_before_creating_an_index(tmp_path, vector):
    class InvalidEncoder(Encoder):
        def embed(self, texts):
            return [vector for text in texts]

    path = tmp_path / "scenes.db"
    with pytest.raises(ValueError, match="finite"):
        SceneDatabase.create(path, pool().blocks, InvalidEncoder())
    assert not path.exists()


def test_hidden_target_in_an_omitted_reference_prevents_publication(tmp_path):
    from test_dataset_builder import Counter, config, corpus, spec

    from endless_voices.dataset.builder import DatasetBuilder

    source = corpus(tmp_path, "mission Current\n\ton offer\n\t\tconversation\n\t\t\t`Target.`\n")
    builder = DatasetBuilder(source, config({"Current": spec()}), Counter())
    leaked = block("other", "Hidden target " * 10000, 4,
                   reference=True, conversation=3, state={})
    builder.sampler.references = lambda *args: [leaked]
    with pytest.raises(ValueError, match="Target source paragraph appears in eligible context"):
        builder.build(tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_local_transformer_encoder_loads_and_masks_padding_without_downloads(tmp_path):
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    from endless_voices.retrieval import TransformersEmbedder

    model_path = tmp_path / "encoder"
    model_path.mkdir()
    vocab = model_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nstarship\narrives\nold\nstory\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(model_path)
    model = BertModel(BertConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=16))
    model.save_pretrained(model_path)
    encoder = TransformersEmbedder.from_pretrained(model_path)
    alone = encoder.embed(["starship"])[0]
    padded = encoder.embed(["starship", "a starship arrives old story"])[0]
    assert len(alone) == 8
    assert torch.allclose(torch.tensor(alone), torch.tensor(padded), atol=1e-6)
    assert encoder.identity == TransformersEmbedder.from_pretrained(model_path).identity
    source = pool()
    with SceneDatabase.create(tmp_path / "scenes.db", source.blocks + source.references,
                              encoder) as database:
        matches = RetrievalSession(database, source).search_context(limit=2)
        assert len(matches) == 2
        assert all(-1.00001 <= match["score"] <= 1.00001 for match in matches)
