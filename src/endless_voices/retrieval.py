"""Search a static scene index through sample-specific eligibility boundaries."""

import json
import math
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Protocol

from endless_voices.context import digest, distances, render

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a scene-writing query, retrieve relevant fictional mission history and "
    "writing passages that support continuity, lore, character voice, and style."
)


class Embedder(Protocol):
    """Encode plain documents and instructed queries in one identified vector space."""

    identity: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class TransformersEmbedder:
    """Encode Qwen3 passages and instructed queries with last-token pooling."""

    def __init__(self, tokenizer, model, model_id, *, max_length=32768, batch_size=4,
                 query_instruction=DEFAULT_QUERY_INSTRUCTION):
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Embedding model identity must be a nonempty string")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("Embedding max_length must be a positive integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Embedding batch_size must be a positive integer")
        if not isinstance(query_instruction, str) or not query_instruction.strip():
            raise ValueError("Query instruction must be nonempty text")
        capacity = getattr(model.config, "max_position_embeddings", max_length)
        if max_length > capacity:
            raise ValueError(f"Embedding limit {max_length} exceeds model capacity {capacity}")
        self.tokenizer, self.model = tokenizer, model.eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.query_instruction = query_instruction
        self.identity = digest({"model": model_id, "pooling": "last-nonpadding-v1",
                                "max_length": max_length, "instruction": query_instruction,
                                "dtype": str(model.dtype)})

    @classmethod
    def from_pretrained(cls, path=DEFAULT_EMBEDDING_MODEL, *, model_id=None,
                        local_files_only=True, device="auto", dtype="auto", max_length=32768,
                        batch_size=4, query_instruction=DEFAULT_QUERY_INSTRUCTION,
                        cache_dir="data/local/hub"):
        """Load a cached Qwen3 encoder on CPU, CUDA or MPS without automatic downloads."""
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        from endless_voices.artifacts import file_hash

        path = Path(path)
        if not path.is_dir():
            path = Path(snapshot_download(str(path), local_files_only=local_files_only,
                                          cache_dir=cache_dir))
        if model_id is None:
            model_id = digest({str(p.relative_to(path)): file_hash(p)
                               for p in sorted(path.rglob("*")) if p.is_file()})
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype == "auto":
            dtype = "float32" if device == "cpu" else "float16"
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  padding_side="left")
        model = AutoModel.from_pretrained(path, local_files_only=True,
                                         dtype=getattr(torch, dtype)).to(device)
        return cls(tokenizer, model, model_id, max_length=max_length, batch_size=batch_size,
                   query_instruction=query_instruction)

    def _embed(self, texts, kind):
        import torch

        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                   truncation=False, return_tensors="pt")
            lengths = batch["attention_mask"].sum(dim=1)
            if (lengths > self.max_length).any():
                longest = int(lengths.max())
                raise ValueError(f"{kind} contains {longest} tokens; embedding limit is "
                                 f"{self.max_length}. Split the passage or shorten the query "
                                 "explicitly; no text was truncated.")
            batch = {key: value.to(self.model.device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"]
                positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
                last = positions.masked_fill(mask == 0, -1).max(dim=1).values
                if (last < 0).any():
                    raise ValueError("Cannot embed an empty token sequence")
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors

    def embed_documents(self, texts):
        """Return normalized passage vectors without adding a query instruction."""
        return self._embed(texts, "Document")

    def embed_queries(self, texts):
        """Return normalized instructed query vectors without dropping query tokens."""
        return self._embed([f"Instruct: {self.query_instruction}\nQuery:{text}" for text in texts],
                           "Instructed query")

    def embed(self, texts):
        """Return plain document vectors for callers of the original encoder interface."""
        return self.embed_documents(texts)


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


def passage_chunks(block):
    """Return complete authored passages tied to an exact eligible parent scene."""
    sid = scene_id(block)
    return [{"id": "passage-" + digest([sid, index, passage]), "scene_id": sid,
             "passage_index": index, "line": passage["line"],
             "text": block["heading"] + "\n" + (
                 "Player choice: " if passage["role"] == "option" else "") + passage["text"]}
            for index, passage in enumerate(block["passages"])]


class SceneDatabase:
    """Read immutable passage vectors; authorization belongs to RetrievalSession."""

    def __init__(self, path, embedder):
        self.embedder = embedder
        self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            metadata = dict(self.connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("format") != "passage-vectors-v2":
                raise ValueError("Unsupported scene index format; rebuild the passage index")
            if metadata.get("embedder") != embedder.identity:
                raise ValueError("Embedding model differs from the scene index")
            self.dimension = int(metadata["dimension"])
        except Exception:
            self.connection.close()
            raise

    @classmethod
    def create(cls, path, blocks, embedder):
        """Persist complete passage vectors without replacing an existing index."""
        path = Path(path)
        if not isinstance(embedder.identity, str) or not embedder.identity:
            raise ValueError("Embedding model identity must be a nonempty string")
        documents = {scene_id(block): deepcopy(block) for block in blocks}
        chunks = [chunk for sid in sorted(documents) for chunk in passage_chunks(documents[sid])]
        if not chunks:
            raise ValueError("Cannot index an empty passage corpus")
        vectors = list(embedder.embed_documents([chunk["text"] for chunk in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError("Embedder returned the wrong number of passage vectors")
        vectors = [_vector(vector) for vector in vectors]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("Passage embedding dimensions differ")
        with path.open("xb"):
            pass
        connection = sqlite3.connect(path)
        try:
            with connection:
                connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                connection.execute("CREATE TABLE scenes (id TEXT PRIMARY KEY, block TEXT)")
                connection.execute("CREATE TABLE passages (id TEXT PRIMARY KEY, scene_id TEXT, "
                                   "chunk TEXT, vector TEXT)")
                connection.execute("CREATE INDEX passage_scenes ON passages (scene_id)")
                connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
                    ("format", "passage-vectors-v2"), ("embedder", embedder.identity),
                    ("dimension", str(dimension)),
                ])
                connection.executemany("INSERT INTO scenes VALUES (?, ?)", [
                    (sid, json.dumps(block, ensure_ascii=False))
                    for sid, block in sorted(documents.items())])
                connection.executemany("INSERT INTO passages VALUES (?, ?, ?, ?)", [
                    (chunk["id"], chunk["scene_id"], json.dumps(chunk, ensure_ascii=False),
                     json.dumps(vector)) for chunk, vector in zip(chunks, vectors, strict=True)])
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
        return cls(path, embedder)

    def close(self):
        """Close the read-only database connection."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _get(self, sid):
        stored = self.connection.execute("SELECT block FROM scenes WHERE id = ?", (sid,)).fetchone()
        if stored is None:
            raise ValueError("Eligible scene is absent from the scene index")
        block = json.loads(stored[0])
        if scene_id(block) != sid:
            raise ValueError("Scene index content differs from its identity")
        expected = {chunk["id"]: chunk for chunk in passage_chunks(block)}
        chunks = {}
        for cid, serialized, vector in self.connection.execute(
                "SELECT id, chunk, vector FROM passages WHERE scene_id = ?", (sid,)):
            chunk = json.loads(serialized)
            if chunk != expected.get(cid):
                raise ValueError("Passage index content differs from its parent scene")
            chunks[cid] = (chunk, _vector(json.loads(vector), self.dimension))
        if chunks.keys() != expected.keys():
            raise ValueError("Scene index is missing an eligible passage")
        return block, chunks


class RetrievalSession:
    """Expose only the interpreted context authorized for one visible encounter."""

    def __init__(self, database, pool):
        self.database = database
        self._pool = deepcopy(pool)
        self._allowed = {}
        self._chunks = {}
        self.trace = []
        for kind, blocks in (("history", self._pool.blocks),
                             ("reference", self._pool.references)):
            for block in blocks:
                sid = scene_id(block)
                stored, chunks = database._get(sid)
                if stored != block:
                    raise ValueError("Scene index does not match the eligible context")
                self._allowed[sid] = (kind, block)
                self._chunks.update(chunks)
        self._distances = distances(self._pool.graph, self._pool.mission)

    def initial_messages(self):
        """Return the fixed instructions and visible encounter without sampled passages."""
        return self._pool.messages(set())

    def _document(self, sid, *, full=False, score=None):
        chunk = self._chunks.get(sid)
        parent_id = chunk[0]["scene_id"] if chunk else sid
        kind, block = self._allowed[parent_id]
        document = {
            "id": sid, "scene_id": parent_id, "kind": kind, "mission": block["mission"],
            "heading": block["heading"], "path": block["path"],
            "lines": [passage["line"] for passage in block["passages"]],
            "source_kind": block.get("source_kind", "conversation"),
        }
        if chunk:
            document["lines"] = [chunk[0]["line"]]
            document["passage_index"] = chunk[0]["passage_index"]
        if kind == "history":
            document["depth"] = self._distances.get(block["mission"])
        else:
            document["usage"] = "Independent writing reference; not current playthrough history."
        text = chunk[0]["text"] if chunk else render(block)
        document["text" if full else "excerpt"] = text
        if score is not None:
            document["score"] = score
        return document

    def _record(self, name, arguments, result):
        entry = {"tool": name, "arguments": deepcopy(arguments),
                 "result_ids": [document["id"] for document in result]}
        matches = [{"scene_id": document["id"], **deepcopy(document["matched_passage"])}
                   for document in result if "matched_passage" in document]
        if matches:
            entry["matched_passages"] = matches
        self.trace.append(entry)
        return result

    def get_history(self, depth):
        """Return complete eligible scenes within a prerequisite distance."""
        if type(depth) is not int or depth < 0:
            raise ValueError("depth must be a nonnegative integer")
        selected = [sid for sid, (kind, block) in self._allowed.items()
                    if kind == "history" and self._distances.get(block["mission"], math.inf)
                    <= depth]
        return self._record("get_history", {"depth": depth},
                            [self._document(sid, full=True) for sid in selected])

    def search_context(self, query=None, kind="all", limit=5, expand_scenes=True):
        """Rank passages and return distinct full scenes, or passages when expansion is disabled."""
        if kind not in {"all", "history", "reference"}:
            raise ValueError("kind must be all, history or reference")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if type(expand_scenes) is not bool:
            raise ValueError("expand_scenes must be a boolean")
        if query is None:
            query = "\n\n".join(message["content"] for message in self._pool.encounter)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty text")
        embedded = list(self.database.embedder.embed_queries([query]))
        if len(embedded) != 1:
            raise ValueError("Embedder returned the wrong number of query vectors")
        vector = _vector(embedded[0], self.database.dimension)
        ranked = sorted(
            ((sum(a * b for a, b in zip(vector, scene_vector, strict=True)), sid)
             for sid, (chunk, scene_vector) in self._chunks.items()
             if kind in {"all", self._allowed[chunk["scene_id"]][0]}),
            key=lambda item: (-item[0], item[1]),
        )
        documents, seen = [], set()
        for score, sid in ranked:
            chunk = self._chunks[sid][0]
            parent_id = chunk["scene_id"]
            if expand_scenes:
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                document = self._document(parent_id, full=True, score=score)
                document["matched_passage"] = {
                    "id": sid, "line": chunk["line"], "passage_index": chunk["passage_index"],
                    "score": score,
                }
            else:
                document = self._document(sid, score=score)
            documents.append(document)
            if len(documents) == limit:
                break
        return self._record("search_context", {
            "query": query, "kind": kind, "limit": limit, "expand_scenes": expand_scenes,
        }, documents)

    def read_context(self, ids):
        """Return complete authorized passages or parent scenes by their IDs."""
        if not isinstance(ids, list) or any(not isinstance(sid, str) for sid in ids):
            raise ValueError("ids must be a list of scene IDs")
        if any(sid not in self._allowed and sid not in self._chunks for sid in ids):
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
                    "Search eligible passages and expand the best matches into complete scenes.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description":
                              "Search text; omit to use only the visible encounter."},
                    "kind": {"type": "string", "enum": ["all", "history", "reference"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100,
                              "description": "Maximum distinct scenes, or passages if expansion "
                                             "is disabled."},
                    "expand_scenes": {"type": "boolean", "default": True,
                                      "description": "Return full scenes with matching passage "
                                                     "evidence; false returns passages only."}},
                    "additionalProperties": False}}},
            {"type": "function", "function": {
                "name": "read_context", "description":
                    "Read passages by id or complete parent scenes by scene_id.",
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
