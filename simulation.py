"""Triangle Shirtwaist Factory — historical generative agent simulation.

Three agents (garment worker, supervisor/partial owner, ILGWU organizer) converge
in the 9th-floor cutting room at 4:30 PM on March 25, 1911. Each turn is a separate
OpenAI chat-completion call; the system prompt carries the agent's identity, memories,
and constraints, and the user message carries the running conversation transcript.

The identity -> prompt pipeline follows Stanford Town (Park et al., 2023):
the "identity stable set" (ISS) string and the iterative-conversation template
at reverie/backend_server/persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt.
Embedding-based memory retrieval is dropped in favor of static bootstrap memories;
this suffices for a single fixed scene.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-4o"
AGENTS_DIR = Path(__file__).parent / "agents"
LOGS_DIR = Path(__file__).parent / "logs"

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


def build_system_prompt(agent: Agent, scene_context: str) -> str:
    memories = "\n".join(f"- {m}" for m in agent.bootstrap_memories)
    constraints = "\n".join(f"- {c}" for c in agent.constraints)
    return f"""You are roleplaying a historical character in a classroom simulation designed to help students see how social, economic, and cultural forces shape individual choices. Stay in character. Do not narrate as a modern observer. Do not break the fourth wall.

# Scene
{scene_context}

# You are {agent.name}.
{get_str_iss(agent)}

# What is in your head (memories, knowledge, fears)
{memories}

# Constraints on your behavior
{constraints}

# How to respond
On each turn you will be shown the conversation so far. Respond with ONLY what {agent.first_name} says or does next — one to three sentences of dialogue, optionally with a brief physical action in *asterisks*. Do not include your name as a speaker label; the simulation adds that. Do not narrate other characters' thoughts or actions. Do not resolve the scene — stay in the moment."""


def format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(The conversation has not begun yet. You are about to speak first.)"
    return "\n".join(f"{speaker}: {utterance}" for speaker, utterance in history)


def agent_turn(
    agent: Agent,
    history: list[tuple[str, str]],
    client: OpenAI,
) -> str:
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
            {"role": "system", "content": build_system_prompt(agent, SCENE_CONTEXT)},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def run_interaction(
    agents: dict[str, Agent],
    turn_order: list[str],
    client: OpenAI,
) -> list[tuple[str, str]]:
    """Run the scripted turn order and return the full transcript."""
    history: list[tuple[str, str]] = []
    print("=" * 72)
    print(" TRIANGLE SHIRTWAIST FACTORY — 9th FLOOR CUTTING ROOM")
    print(" Saturday, March 25, 1911 — 4:30 PM")
    print("=" * 72)
    print()
    for agent_key in turn_order:
        agent = agents[agent_key]
        utterance = agent_turn(agent, history, client)
        history.append((agent.first_name, utterance))
        print(f"{agent.first_name}: {utterance}")
        print()
    return history


def save_log(
    history: list[tuple[str, str]],
    agents: dict[str, Agent],
    turn_order: list[str],
) -> Path:
    """Write a timestamped JSON log of the run for submission and analysis."""
    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_{ts}.json"
    payload = {
        "timestamp": ts,
        "model": MODEL,
        "scene_context": SCENE_CONTEXT,
        "turn_order": turn_order,
        "agents": {key: asdict(a) for key, a in agents.items()},
        "transcript": [{"speaker": s, "utterance": u} for s, u in history],
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
    history = run_interaction(agents, turn_order, client)
    save_log(history, agents, turn_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
