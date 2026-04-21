"""Triangle Shirtwaist Factory — historical generative agent simulation.

Three agents (garment worker, supervisor/partial owner, ILGWU organizer) converge
in the 9th-floor cutting room at 4:30 PM on March 25, 1911. Each turn is a separate
OpenAI chat-completion call; the system prompt carries the agent's identity and
retrieved memories, and the user message carries the running conversation transcript.

Agent architecture follows Stanford Town (Park et al., 2023):

  Identity (ISS) -----+
                      |
  Bootstrap memories -+--> AssociativeMemory  --retrieve(query, k)--> top-k
                      |                                                   |
  Observations -------+                                                   v
  (each utterance                                                    System prompt
   other agents hear)                                                     |
                                                                          v
                                                                   chat.completions

Each agent owns an AssociativeMemory (memory.py). Bootstrap memories are seeded
at load; every utterance heard by an agent becomes an observation added to their
stream. On each turn, before generating the next utterance, the agent retrieves
top-k memories scored by recency + relevance + importance, and only those go
into the system prompt (not the full bootstrap list).

When accumulated observation-importance crosses a threshold, the agent reflects
at the top of their next turn: the model produces a small number of first-person
insights from recent memories, and those insights are written back into the
memory stream as new reflection nodes. Reflection fires *before* retrieval for
that same turn, so fresh insights are eligible to be retrieved into the prompt.

See memory.py for the retrieval + reflection implementations and their
correspondences to Stanford Town's `cognitive_modules/retrieve.py`,
`cognitive_modules/reflect.py`, and `memory_structures/associative_memory.py`.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from memory import AssociativeMemory


MODEL = "gpt-4o"
AGENTS_DIR = Path(__file__).parent / "agents"
LOGS_DIR = Path(__file__).parent / "logs"
RETRIEVAL_K = 5  # memories pulled into the prompt per turn

SCENE_CONTEXT = """\
It is Saturday, March 25, 1911, at 4:30 PM, in New York City.

The scene is the 9th-floor cutting room of the Triangle Shirtwaist Company, in
the Asch Building at the corner of Washington Place and Greene Street. The floor
is loud with sewing machines on the 8th floor below and the hum of the cutting
tables here. Piles of cotton shirtwaist cuttings are heaped under the tables;
the air is thick with lint. The Washington Place stairwell door is locked, as
it is every shift; the Greene Street freight door is the only open exit. About
four hundred workers are on the floor, mostly young immigrant women. The shift
ends at 5:00 PM.

Leonora Russo is at her cutting table. Rosa Peretz, an ILGWU organizer banned
from the building by name, has slipped in through the freight entrance under a
shawl. She approaches Leonora. Max Bernstein, the floor supervisor and a partial
owner, is walking the floor and will shortly notice Rosa.

None of the three knows that in approximately thirty minutes, a scrap bin on
the 8th floor will ignite and the fire will spread through this building,
killing 146 workers. They are acting in the moment, with the knowledge and
fears available to each of them on this ordinary late-Saturday afternoon.
"""


@dataclass
class Agent:
    name: str
    first_name: str
    last_name: str
    age: int
    innate: str
    learned: str
    currently: str
    daily_plan_req: str
    bootstrap_memories: list[str]
    constraints: list[str]


def load_agent(path: Path) -> Agent:
    with open(path) as f:
        data = json.load(f)
    return Agent(**data)


def get_str_iss(agent: Agent) -> str:
    """Identity Stable Set — Stanford Town's get_str_iss() pattern.

    See Stanford_Town/repo/reverie/backend_server/persona/memory_structures/scratch.py:382
    """
    return (
        f"Name: {agent.name}\n"
        f"Age: {agent.age}\n"
        f"Innate traits: {agent.innate}\n"
        f"Learned traits: {agent.learned}\n"
        f"Currently: {agent.currently}\n"
        f"Daily plan requirement: {agent.daily_plan_req}\n"
        f"Current date: Saturday, March 25, 1911\n"
    )


def build_system_prompt(
    agent: Agent,
    scene_context: str,
    retrieved_memories: list[str],
) -> str:
    memories = "\n".join(f"- {m}" for m in retrieved_memories)
    constraints = "\n".join(f"- {c}" for c in agent.constraints)
    return f"""You are roleplaying a historical character in a classroom simulation designed to help students see how social, economic, and cultural forces shape individual choices. Stay in character. Do not narrate as a modern observer. Do not break the fourth wall.

# Scene
{scene_context}

# You are {agent.name}.
{get_str_iss(agent)}

# What is in your head right now (the memories most relevant to this moment)
{memories}

# Constraints on your behavior
{constraints}

# How to respond
On each turn you will be shown the conversation so far. Respond with ONLY what {agent.first_name} says or does next — one to three sentences of dialogue, optionally with a brief physical action in *asterisks*. Do not include your name as a speaker label; the simulation adds that. Do not narrate other characters' thoughts or actions. Do not resolve the scene — stay in the moment."""


def format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(The conversation has not begun yet. You are about to speak first.)"
    return "\n".join(f"{speaker}: {utterance}" for speaker, utterance in history)


def build_retrieval_query(
    agent: Agent,
    history: list[tuple[str, str]],
) -> str:
    """Form a retrieval query from the agent's current goal plus the last couple
    of utterances they've heard. This concatenation is a simpler alternative to
    Stanford Town's multi-focal-point retrieval in new_retrieve()."""
    recent = " ".join(u for _, u in history[-2:])
    return f"{agent.currently} {recent}".strip()


def agent_turn(
    agent: Agent,
    memory: AssociativeMemory,
    history: list[tuple[str, str]],
    turn_num: int,
    client: OpenAI,
) -> tuple[str, list[str]]:
    """Generate one agent turn with retrieval. Returns (utterance, retrieved_memory_strings)."""
    query = build_retrieval_query(agent, history)
    retrieved_nodes = memory.retrieve(query, current_turn=turn_num, k=RETRIEVAL_K)
    retrieved_strs = [n.content for n in retrieved_nodes]

    user_message = (
        f"Conversation so far on the 9th-floor cutting room:\n\n"
        f"{format_history(history)}\n\n"
        f"What does {agent.first_name} say or do next? Remember: one to three sentences, "
        f"in character, in period voice. No speaker label."
    )
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": build_system_prompt(agent, SCENE_CONTEXT, retrieved_strs),
            },
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip(), retrieved_strs


def run_interaction(
    agents: dict[str, Agent],
    memories: dict[str, AssociativeMemory],
    turn_order: list[str],
    client: OpenAI,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Run the scripted turn order. Return (transcript, per_turn_log).

    per_turn_log[i] contains the query, the retrieved memory contents, and the
    utterance produced — enough to audit what the retrieval module did.
    """
    history: list[tuple[str, str]] = []
    per_turn: list[dict] = []

    print("=" * 72)
    print(" TRIANGLE SHIRTWAIST FACTORY — 9th FLOOR CUTTING ROOM")
    print(" Saturday, March 25, 1911 — 4:30 PM")
    print("=" * 72)
    print()

    for turn_num, agent_key in enumerate(turn_order):
        agent = agents[agent_key]
        memory = memories[agent_key]

        # If enough observation-importance has accumulated since the last
        # reflection, fire reflection BEFORE retrieval so the fresh insights
        # are eligible to be retrieved into this same turn's system prompt.
        reflection_log: dict | None = None
        if memory.should_reflect():
            new_insights = memory.reflect(current_turn=turn_num)
            if new_insights:
                insight_strs = [n.content for n in new_insights]
                reflection_log = {
                    "triggered": True,
                    "insights": insight_strs,
                }
                print(f"  [{agent.first_name} reflects]")
                for s in insight_strs:
                    print(f"    - {s}")

        query = build_retrieval_query(agent, history)
        utterance, retrieved = agent_turn(
            agent, memory, history, turn_num, client
        )
        history.append((agent.first_name, utterance))

        # Every OTHER agent in the room hears the utterance and records it as
        # an observation in their own memory stream, with importance rated by
        # the model. The speaker does not record their own speech — the
        # conversation history passed to the prompt already contains it.
        for other_key, other_memory in memories.items():
            if other_key == agent_key:
                continue
            observation = f'{agent.name} said: "{utterance}"'
            other_memory.add_observation(
                content=observation,
                speaker=agent.name,
                turn=turn_num,
                rate_with_model=True,
            )

        per_turn.append(
            {
                "turn": turn_num,
                "speaker": agent.first_name,
                "reflection": reflection_log,
                "retrieval_query": query,
                "retrieved_memories": retrieved,
                "utterance": utterance,
            }
        )

        print(f"{agent.first_name}: {utterance}")
        print()

    return history, per_turn


def save_log(
    history: list[tuple[str, str]],
    per_turn: list[dict],
    agents: dict[str, Agent],
    memories: dict[str, AssociativeMemory],
    turn_order: list[str],
) -> Path:
    """Write a timestamped JSON log of the run for submission and analysis.

    The log captures: scene, turn order, per-agent identity, per-turn retrieval
    (query + top-k memory contents), per-turn reflection audit (when reflection
    fires, the insights it produced), transcript, and each agent's final
    memory stream (with reflection nodes visible as source="reflection").
    Embeddings are elided to keep the file human-readable.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_{ts}.json"
    payload = {
        "timestamp": ts,
        "model": MODEL,
        "retrieval_k": RETRIEVAL_K,
        "scene_context": SCENE_CONTEXT,
        "turn_order": turn_order,
        "agents": {key: asdict(a) for key, a in agents.items()},
        "per_turn": per_turn,
        "transcript": [{"speaker": s, "utterance": u} for s, u in history],
        "final_memory_streams": {
            key: memory.dump(include_embeddings=False)
            for key, memory in memories.items()
        },
    }
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[log saved: {log_path.relative_to(Path(__file__).parent)}]")
    return log_path


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. Run:\n"
            '    export OPENAI_API_KEY="sk-..."',
            file=sys.stderr,
        )
        return 1

    agents = {
        "leonora": load_agent(AGENTS_DIR / "leonora_russo.json"),
        "max": load_agent(AGENTS_DIR / "max_bernstein.json"),
        "rosa": load_agent(AGENTS_DIR / "rosa_peretz.json"),
    }

    # Scripted turn order: Rosa approaches Leonora, they exchange a few lines,
    # Max notices Rosa and intervenes, all three negotiate the confrontation.
    turn_order = [
        "rosa",      # Rosa opens: quiet approach, names the union and the ask
        "leonora",   # Leonora: hesitant, afraid
        "rosa",      # Rosa: specifics — locked doors, sprinklers
        "leonora",   # Leonora: the family, the sister
        "max",       # Max: recognizes Rosa, confronts
        "rosa",      # Rosa: holds ground
        "max",       # Max: the self-made-man frame, orders her out
        "leonora",   # Leonora: caught between, tries to disappear
        "rosa",      # Rosa: last word to Leonora
        "max",       # Max: closes the scene
    ]

    client = OpenAI()

    # Build one AssociativeMemory per agent; seed with bootstrap memories.
    print("[seeding agent memory streams with bootstrap memories…]")
    memories = {key: AssociativeMemory(client, a.name) for key, a in agents.items()}
    for key, a in agents.items():
        memories[key].seed_bootstrap(a.bootstrap_memories)
        print(f"  {key}: {len(memories[key].nodes)} bootstrap nodes embedded")
    print()

    history, per_turn = run_interaction(agents, memories, turn_order, client)
    save_log(history, per_turn, agents, memories, turn_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
