"""Associative memory with retrieval — a faithful port of Stanford Town's
`memory_structures/associative_memory.py` + `cognitive_modules/retrieve.py`,
simplified for a single-scene three-agent simulation.

Each agent owns an AssociativeMemory. Bootstrap memories are seeded at load.
Every conversational utterance heard by an agent is added to that agent's
memory stream as an observation node. On each turn, before generating a
response, the speaking agent retrieves the top-k most relevant memories via
a weighted combination of recency, relevance, and importance — and only those
retrieved memories are injected into the system prompt.

Scoring follows Park et al. (2023) §A.1.1:

    score = W_RECENCY * recency_norm
          + W_RELEVANCE * relevance_norm
          + W_IMPORTANCE * importance_norm

where:
- recency  = RECENCY_DECAY ** (current_turn - last_accessed_turn)
- relevance = cosine(query_embedding, node_embedding)
- importance = node.importance / 10    (1-10 scale, rated at creation)

Each of the three is min-max normalized across the candidate set before
weighting. Defaults mirror Stanford Town's `scratch.json` (recency_w=1,
relevance_w=1, importance_w=1, recency_decay=0.995).

Differences from Stanford Town, documented for honesty:
- Memory nodes are single-field `content` strings. Stanford Town splits each
  node into (subject, predicate, object) + description; that structure is
  used by their reflect module and for spatial/event memories. Unnecessary
  here.
- `keywords` and `kw_strength` are not implemented; retrieval uses embeddings
  only. Stanford Town also uses keyword strength as a secondary signal and
  as a reflection trigger.
- A single query string is used per retrieval. Stanford Town's `new_retrieve`
  takes a list of focal points and unions the results; our simpler version
  concatenates recent context + the agent's `currently` into one query.
- Agents do not record their own utterances as observations. The conversation
  history is passed to the model directly, so self-observation would be
  redundant. Park et al. do record self-speech for cross-day continuity,
  which doesn't matter in a one-scene simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

from openai import OpenAI


EMBED_MODEL = "text-embedding-3-small"
IMPORTANCE_MODEL = "gpt-4o-mini"

RECENCY_DECAY = 0.995
W_RECENCY = 1.0
W_RELEVANCE = 1.0
W_IMPORTANCE = 1.0

BOOTSTRAP_IMPORTANCE = 6  # background memories: middling-high by default


@dataclass
class MemoryNode:
    content: str
    source: str                 # "bootstrap" | "observation"
    created_turn: int           # -1 for bootstrap; otherwise the turn at which it was heard
    last_accessed_turn: int
    importance: int             # 1-10
    speaker: str | None = None  # for observations: who said it; None for bootstrap
    embedding: list[float] = field(default_factory=list, repr=False)

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_embedding:
            d["embedding"] = f"[{len(self.embedding)}-dim vector]"
        return d


def embed(client: OpenAI, text: str) -> list[float]:
    """Return the embedding vector for `text`."""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rate_importance(client: OpenAI, content: str, agent_name: str) -> int:
    """Ask the model to rate the poignancy of `content` for `agent_name` (1-10).

    Port of Stanford Town's `generate_poig_score()` in run_gpt_prompt.py.
    Uses a small/cheap model — the rating is a scalar, not a dialog turn.
    """
    prompt = (
        f"On a scale of 1 to 10, rate the likely poignancy of the following "
        f"event for {agent_name}. 1 = completely mundane (brushing teeth, "
        f"checking the time). 10 = life-altering (a death, a marriage, "
        f"losing one's job). Respond with ONLY a single integer 1-10, "
        f"nothing else.\n\n"
        f"Event: {content}\n\n"
        f"Rating:"
    )
    response = client.chat.completions.create(
        model=IMPORTANCE_MODEL,
        max_tokens=3,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    try:
        value = int("".join(c for c in text if c.isdigit())[:2])
        return max(1, min(10, value))
    except (ValueError, TypeError):
        return 5


class AssociativeMemory:
    """An agent's memory stream, with embedding-based retrieval."""

    def __init__(self, client: OpenAI, owner_name: str):
        self.client = client
        self.owner = owner_name
        self.nodes: list[MemoryNode] = []

    def seed_bootstrap(self, bootstrap_memories: list[str]) -> None:
        """Seed the memory stream with the agent's bootstrap memories.
        Bootstrap memories all get embeddings and importance=BOOTSTRAP_IMPORTANCE."""
        for m in bootstrap_memories:
            node = MemoryNode(
                content=m,
                source="bootstrap",
                created_turn=-1,
                last_accessed_turn=-1,
                importance=BOOTSTRAP_IMPORTANCE,
                speaker=None,
                embedding=embed(self.client, m),
            )
            self.nodes.append(node)

    def add_observation(
        self,
        content: str,
        speaker: str,
        turn: int,
        rate_with_model: bool = True,
    ) -> MemoryNode:
        """Write a new observation into the stream. Importance is rated by the
        model by default; pass rate_with_model=False to skip that API call
        and use a default importance of 5."""
        importance = (
            rate_importance(self.client, content, self.owner)
            if rate_with_model
            else 5
        )
        node = MemoryNode(
            content=content,
            source="observation",
            created_turn=turn,
            last_accessed_turn=turn,
            importance=importance,
            speaker=speaker,
            embedding=embed(self.client, content),
        )
        self.nodes.append(node)
        return node

    def retrieve(
        self,
        query: str,
        current_turn: int,
        k: int = 5,
    ) -> list[MemoryNode]:
        """Return the top-k memory nodes for `query` at `current_turn`.

        Score = w_recency * recency_norm
              + w_relevance * relevance_norm
              + w_importance * importance_norm
        Each component is min-max normalized across all nodes.
        Side effect: retrieved nodes have their last_accessed_turn updated,
        so that frequently-retrieved memories decay more slowly. This matches
        Stanford Town.
        """
        if not self.nodes:
            return []
        q_vec = embed(self.client, query)

        recencies = [
            RECENCY_DECAY ** max(0, current_turn - n.last_accessed_turn)
            for n in self.nodes
        ]
        relevances = [cosine(q_vec, n.embedding) for n in self.nodes]
        importances = [n.importance / 10.0 for n in self.nodes]

        rec_n = _normalize(recencies)
        rel_n = _normalize(relevances)
        imp_n = _normalize(importances)

        scored = [
            (
                W_RECENCY * rec_n[i]
                + W_RELEVANCE * rel_n[i]
                + W_IMPORTANCE * imp_n[i],
                i,
            )
            for i in range(len(self.nodes))
        ]
        scored.sort(reverse=True)

        top_indices = [i for _, i in scored[:k]]
        top_nodes = [self.nodes[i] for i in top_indices]

        for n in top_nodes:
            n.last_accessed_turn = current_turn

        return top_nodes

    def dump(self, include_embeddings: bool = False) -> list[dict[str, Any]]:
        """Serialize the memory stream for logging."""
        return [n.to_dict(include_embedding=include_embeddings) for n in self.nodes]
