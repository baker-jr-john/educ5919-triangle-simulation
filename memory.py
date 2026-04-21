"""Associative memory with retrieval + reflection — a faithful port of Stanford
Town's `memory_structures/associative_memory.py` + `cognitive_modules/retrieve.py`
+ `cognitive_modules/reflect.py`, simplified for a single-scene three-agent
simulation.

Each agent owns an AssociativeMemory. Bootstrap memories are seeded at load.
Every conversational utterance heard by an agent is added to that agent's
memory stream as an observation node. On each turn, before generating a
response, the speaking agent retrieves the top-k most relevant memories via
a weighted combination of recency, relevance, and importance — and only those
retrieved memories are injected into the system prompt.

Between observations, when the accumulated importance of new observations
crosses a threshold, the agent **reflects**: the model is asked to draw a
small number of high-level first-person insights from recent memories, and
those insights are written back into the stream as new nodes with source
"reflection" and elevated importance. Because reflections live in the same
memory stream as bootstrap + observations, they become candidates for future
retrieval, which lets later turns reason over second-order understandings
("this man is not reachable through argument") rather than raw events alone.

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
- Reflection uses most-recent-N as its input pool rather than generating
  focal questions and retrieving per question. For a ten-turn scene, "most
  recent" is close enough to the ideal pool that the extra generate-questions
  round trip is not worth it. Stanford Town's threshold of 150 accumulated
  importance targets multi-day runs; we use 18 so reflection fires once or
  twice per agent in this scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

from openai import OpenAI


EMBED_MODEL = "text-embedding-3-small"
IMPORTANCE_MODEL = "gpt-4o-mini"
REFLECTION_MODEL = "gpt-4o"

RECENCY_DECAY = 0.995
W_RECENCY = 1.0
W_RELEVANCE = 1.0
W_IMPORTANCE = 1.0

BOOTSTRAP_IMPORTANCE = 6  # background memories: middling-high by default

# Reflection tuning. Accumulated observation-importance since the last reflection
# is subtracted from importance_trigger; when it hits 0, reflection fires and
# the trigger resets. Stanford Town uses a threshold of 150 for multi-day runs;
# our scene is 10 turns, so we use a much smaller threshold. REFLECTION_THRESHOLD
# of 18 fires reflection after roughly 3 observations of middling importance,
# so each agent reflects once or twice in a ten-turn scene.
REFLECTION_THRESHOLD = 18
REFLECTION_RECENT_N = 10   # how many recent non-reflection nodes feed the insight prompt
REFLECTION_N_INSIGHTS = 2  # how many insights to write back per trigger
REFLECTION_IMPORTANCE = 8  # reflections are weightier than observations by default


@dataclass
class MemoryNode:
    content: str
    source: str                 # "bootstrap" | "observation" | "reflection"
    created_turn: int           # -1 for bootstrap; otherwise the turn at which it was created
    last_accessed_turn: int
    importance: int             # 1-10
    speaker: str | None = None  # for observations: who said it; for reflections: the reflecting agent; None for bootstrap
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


def generate_reflection_insights(
    client: OpenAI,
    agent_name: str,
    recent_memories: list[str],
    n_insights: int = REFLECTION_N_INSIGHTS,
) -> list[str]:
    """Ask the model to draw high-level first-person insights from a batch of
    recent memories. Port of Stanford Town's reflection prompt in
    cognitive_modules/reflect.py — simplified: we skip the focal-question step
    (Park et al. first generate questions, then retrieve per question) because
    the scene is short enough that most-recent-N is a reasonable proxy for the
    reflection pool.
    """
    memories_text = "\n".join(f"- {m}" for m in recent_memories)
    prompt = (
        f"You are {agent_name}. You have been living through a tense situation "
        f"and accumulating observations. Look at the recent events below and draw "
        f"{n_insights} high-level insights — things you are starting to understand "
        f"about what is happening, about the other people in the room, or about "
        f"your own position. Each insight should be one sentence, first-person "
        f"(\"I see that...\", \"I am beginning to realize...\", \"It is becoming "
        f"clear to me that...\"). Do not speculate beyond what the observations "
        f"support. Stay in your character's voice and period. Return ONLY the "
        f"insights, one per line, with no numbering, bullets, or preamble.\n\n"
        f"Recent events:\n{memories_text}\n\n"
        f"Insights:"
    )
    response = client.chat.completions.create(
        model=REFLECTION_MODEL,
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    insights: list[str] = []
    for line in raw.split("\n"):
        cleaned = line.strip().lstrip("-*•0123456789.) ").strip()
        if cleaned:
            insights.append(cleaned)
    return insights[:n_insights]


class AssociativeMemory:
    """An agent's memory stream, with embedding-based retrieval."""

    def __init__(self, client: OpenAI, owner_name: str):
        self.client = client
        self.owner = owner_name
        self.nodes: list[MemoryNode] = []
        # Accumulated-importance countdown. Decremented by each observation's
        # importance; when it reaches 0 or below, reflect() is due to fire and
        # the trigger resets. Bootstrap seeding does NOT decrement the trigger —
        # those are the agent's prior life, not fresh experience.
        self.importance_trigger: int = REFLECTION_THRESHOLD

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
        self.importance_trigger -= importance
        return node

    def should_reflect(self) -> bool:
        """True when accumulated observation importance has crossed the threshold."""
        return self.importance_trigger <= 0

    def reflect(self, current_turn: int) -> list[MemoryNode]:
        """Generate high-level insights from recent memories and write them back
        as new reflection nodes. Resets the importance_trigger. Returns the
        newly-created reflection nodes (so the caller can log what was written).

        Port of Stanford Town's `cognitive_modules/reflect.py`, simplified:
        - Uses most-recent-N memories as the reflection pool rather than
          generating focal questions and retrieving per question. For a
          ten-turn single-scene simulation, most-recent is a reasonable proxy.
        - Does not build a hierarchy of reflections-of-reflections. Stanford
          Town treats reflections as first-class candidates for future
          reflection pools; here we exclude reflections from their own pool
          to keep output grounded in observations.
        """
        pool_sources = {"bootstrap", "observation"}
        recent = [n for n in self.nodes if n.source in pool_sources][-REFLECTION_RECENT_N:]
        if not recent:
            self.importance_trigger = REFLECTION_THRESHOLD
            return []

        insights = generate_reflection_insights(
            self.client,
            self.owner,
            [n.content for n in recent],
            n_insights=REFLECTION_N_INSIGHTS,
        )
        new_nodes: list[MemoryNode] = []
        for insight in insights:
            node = MemoryNode(
                content=insight,
                source="reflection",
                created_turn=current_turn,
                last_accessed_turn=current_turn,
                importance=REFLECTION_IMPORTANCE,
                speaker=self.owner,
                embedding=embed(self.client, insight),
            )
            self.nodes.append(node)
            new_nodes.append(node)

        self.importance_trigger = REFLECTION_THRESHOLD
        return new_nodes

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
