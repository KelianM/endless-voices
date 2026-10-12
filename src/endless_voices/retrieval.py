"""Search a static scene index through sample-specific eligibility boundaries."""

import json
import math
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Protocol

from endless_voices.context import digest, distances, render


class Embedder(Protocol):
    """Encode texts independently in one identified vector space."""

    identity: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class TransformersEmbedder:
    """Mean-pool a caller-supplied local encoder into scene vectors."""

    def __init__(self, tokenizer, model, model_id, *, max_length=512, batch_size=16):
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Embedding model identity must be a nonempty string")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("Embedding max_length must be a positive integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Embedding batch_size must be a positive integer")
        self.tokenizer, self.model = tokenizer, model.eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.identity = f"{model_id}:masked-mean-v1:max-length={max_length}"

    @classmethod
    def from_pretrained(cls, path, *, model_id=None, local_files_only=True):
        """Load an existing local encoder and identify its files by content."""
        from transformers import AutoModel, AutoTokenizer

        from endless_voices.artifacts import file_hash

        path = Path(path)
        if model_id is None:
            if not path.is_dir():
                raise ValueError("A local model directory or explicit model_id is required")
            model_id = digest({str(p.relative_to(path)): file_hash(p)
                               for p in sorted(path.rglob("*")) if p.is_file()})
        return cls(
            AutoTokenizer.from_pretrained(path, local_files_only=local_files_only),
            AutoModel.from_pretrained(path, local_files_only=local_files_only), model_id,
        )

    def embed(self, texts):
        """Return mean-pooled vectors without fitting on the indexed corpus."""
        import torch

        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                   truncation=True, max_length=self.max_length,
                                   return_tensors="pt")
            batch = {key: value.to(self.model.device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.extend(pooled.float().cpu().tolist())
        return vectors


def scene_id(block):
    """Identify an exact interpreted scene, including its scoped state and wording."""
    return "scene-" + digest(block)


def _vector(values, dimension=None):
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("Embedding must contain finite numbers") from error
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding must contain finite numbers")
    if dimension is not None and len(vector) != dimension:
        raise ValueError("Embedding dimension differs from the scene index")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must have a finite nonzero norm")
    return [value / norm for value in vector]


class SceneDatabase:
    """Read immutable scene vectors; authorization belongs to RetrievalSession."""

    def __init__(self, path, embedder):
        self.embedder = embedder
        self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            metadata = dict(self.connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("format") != "scene-vectors-v1":
                raise ValueError("Unsupported scene index format")
            if metadata.get("embedder") != embedder.identity:
                raise ValueError("Embedding model differs from the scene index")
            self.dimension = int(metadata["dimension"])
        except Exception:
            self.connection.close()
            raise

    @classmethod
    def create(cls, path, blocks, embedder):
        """Persist vectors for exact eligible scene variants without replacing an index."""
        path = Path(path)
        if not isinstance(embedder.identity, str) or not embedder.identity:
            raise ValueError("Embedding model identity must be a nonempty string")
        documents = {scene_id(block): deepcopy(block) for block in blocks}
        if not documents:
            raise ValueError("Cannot index an empty scene corpus")
        ids = sorted(documents)
        vectors = list(embedder.embed([render(documents[sid]) for sid in ids]))
        if len(vectors) != len(ids):
            raise ValueError("Embedder returned the wrong number of scene vectors")
        vectors = [_vector(vector) for vector in vectors]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("Scene embedding dimensions differ")
        # Exclusive creation prevents an accidental overwrite of a published index.
        with path.open("xb"):
            pass
        try:
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                connection.execute(
                    "CREATE TABLE scenes (id TEXT PRIMARY KEY, block TEXT, vector TEXT)")
                connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
                    ("format", "scene-vectors-v1"), ("embedder", embedder.identity),
                    ("dimension", str(dimension)),
                ])
                connection.executemany("INSERT INTO scenes VALUES (?, ?, ?)", [
                    (sid, json.dumps(documents[sid], ensure_ascii=False), json.dumps(vector))
                    for sid, vector in zip(ids, vectors, strict=True)
                ])
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return cls(path, embedder)

    def close(self):
        """Close the read-only database connection."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _get(self, sid):
        row = self.connection.execute("SELECT block, vector FROM scenes WHERE id = ?", (sid,))
        stored = row.fetchone()
        if stored is None:
            raise ValueError("Eligible scene is absent from the scene index")
        block, vector = json.loads(stored[0]), json.loads(stored[1])
        if scene_id(block) != sid:
            raise ValueError("Scene index content differs from its identity")
        return block, _vector(vector, self.dimension)


class RetrievalSession:
    """Expose only the interpreted context authorized for one visible encounter."""

    def __init__(self, database, pool):
        self.database = database
        self._pool = deepcopy(pool)
        self._allowed = {}
        self.trace = []
        for kind, blocks in (("history", self._pool.blocks),
                             ("reference", self._pool.references)):
            for block in blocks:
                sid = scene_id(block)
                stored, vector = database._get(sid)
                if stored != block:
                    raise ValueError("Scene index does not match the eligible context")
                self._allowed[sid] = (kind, block, vector)
        self._distances = distances(self._pool.graph, self._pool.mission)

    def initial_messages(self):
        """Return the fixed instructions and visible encounter without sampled passages."""
        return self._pool.messages(set())

    def _document(self, sid, *, full=False, score=None):
        kind, block, _ = self._allowed[sid]
        document = {
            "id": sid, "kind": kind, "mission": block["mission"],
            "heading": block["heading"], "path": block["path"],
            "lines": [passage["line"] for passage in block["passages"]],
            "source_kind": block.get("source_kind", "conversation"),
        }
        if kind == "history":
            document["depth"] = self._distances.get(block["mission"])
        else:
            document["usage"] = "Independent writing reference; not current playthrough history."
        document["text" if full else "excerpt"] = render(block) if full else render(block)[:320]
        if score is not None:
            document["score"] = score
        return document

    def _record(self, name, arguments, result):
        self.trace.append({"tool": name, "arguments": deepcopy(arguments),
                           "result_ids": [document["id"] for document in result]})
        return result

    def get_history(self, depth):
        """Return complete eligible scenes within a prerequisite distance."""
        if type(depth) is not int or depth < 0:
            raise ValueError("depth must be a nonnegative integer")
        selected = [sid for sid, (kind, block, _) in self._allowed.items()
                    if kind == "history" and self._distances.get(block["mission"], math.inf)
                    <= depth]
        return self._record("get_history", {"depth": depth},
                            [self._document(sid, full=True) for sid in selected])

    def search_context(self, query=None, kind="all", limit=5):
        """Rank eligible scene vectors using a supplied query or the visible encounter."""
        if kind not in {"all", "history", "reference"}:
            raise ValueError("kind must be all, history or reference")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if query is None:
            query = "\n\n".join(message["content"] for message in self._pool.encounter)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty text")
        embedded = list(self.database.embedder.embed([query]))
        if len(embedded) != 1:
            raise ValueError("Embedder returned the wrong number of query vectors")
        vector = _vector(embedded[0], self.database.dimension)
        ranked = sorted(
            ((sum(a * b for a, b in zip(vector, scene_vector, strict=True)), sid)
             for sid, (scene_kind, _, scene_vector) in self._allowed.items()
             if kind in {"all", scene_kind}),
            key=lambda item: (-item[0], item[1]),
        )[:limit]
        return self._record("search_context", {"query": query, "kind": kind, "limit": limit},
                            [self._document(sid, score=score) for score, sid in ranked])

    def read_context(self, ids):
        """Return complete authorized scenes, rejecting unknown and ineligible IDs alike."""
        if not isinstance(ids, list) or any(not isinstance(sid, str) for sid in ids):
            raise ValueError("ids must be a list of scene IDs")
        if any(sid not in self._allowed for sid in ids):
            raise ValueError("Scene is not eligible for this encounter")
        return self._record("read_context", {"ids": ids},
                            [self._document(sid, full=True) for sid in ids])

    @property
    def tool_schemas(self):
        """Return JSON function schemas suitable for a model tool interface."""
        return [
            {"type": "function", "function": {
                "name": "get_history", "description": "Read eligible mission history by depth.",
                "parameters": {"type": "object", "properties": {
                    "depth": {"type": "integer", "minimum": 0}},
                    "required": ["depth"], "additionalProperties": False}}},
            {"type": "function", "function": {
                "name": "search_context", "description":
                    "Search eligible history and independent writing references by meaning.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description":
                              "Search text; omit to use only the visible encounter."},
                    "kind": {"type": "string", "enum": ["all", "history", "reference"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                    "additionalProperties": False}}},
            {"type": "function", "function": {
                "name": "read_context", "description": "Read eligible scenes by returned IDs.",
                "parameters": {"type": "object", "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["ids"], "additionalProperties": False}}},
        ]

    def call_tool(self, name, arguments):
        """Dispatch one parsed tool call without exposing database-wide access."""
        methods = {"get_history": self.get_history, "search_context": self.search_context,
                   "read_context": self.read_context}
        if name not in methods or not isinstance(arguments, dict):
            raise ValueError("Unknown context tool or invalid arguments")
        try:
            return methods[name](**arguments)
        except TypeError as error:
            raise ValueError("Invalid context tool arguments") from error
