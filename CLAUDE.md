# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDUC_5919 course project implementing a historical generative agent simulation of the Triangle Shirtwaist Factory (NYC, March 25, 1911). Three LLM-powered agents — a garment worker, a factory supervisor/partial owner, and a union organizer — interact in a historically constrained scene to make systemic labor dynamics visible to learners.

See `simulation_plan.md` for the design (historical context, agent identities, learning objective, key interaction) and `README.md` for the architecture walkthrough and adaptation guide.

## Commands

### Set up the environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the simulation
```bash
export OPENAI_API_KEY="sk-..."
python simulation.py
```

Each run writes `logs/run_<timestamp>.json` with per-turn retrieval + reflection audits and each agent's final memory stream. A full 10-turn run costs roughly $0.15.

### Run the Jupyter notebook version
```bash
jupyter notebook simulation.ipynb
```

### Regenerate the report PDF
```bash
pandoc report.md -o report.pdf --pdf-engine=weasyprint
```
`report.md` is the authoritative source; `report.pdf` is the submission artifact.

## Architecture

### Agent data (`agents/*.json`)
Each agent file mirrors the Stanford Town `scratch.json` pattern with fields:
- `innate` / `learned` / `currently` — layered identity (permanent traits → stable background → current goal)
- `bootstrap_memories` — first-person memories seeded at simulation start (what the agent knows, has seen, fears); each is embedded at load and inserted into the agent's memory stream
- `constraints` — explicit behavioral constraints passed into the system prompt to keep responses historically grounded
- `daily_plan_req` — the agent's standing purpose, included in the ISS string

### Memory + cognition (`memory.py`)
Port of Stanford Town's cognitive modules, scoped down for a single-scene conversation:

- **`AssociativeMemory`** — node stream (bootstrap + observation + reflection nodes). Each node carries `content`, embedding, `importance` (1–10), `created_turn`, `last_accessed_turn`, `source`, and `speaker`. Ports `memory_structures/associative_memory.py`.
- **`retrieve(query, current_turn, k)`** — min-max normalized `recency + relevance + importance` (recency decay 0.995, all weights 1.0), returns top-k, bumps `last_accessed_turn`. Ports `cognitive_modules/retrieve.py`.
- **`add_observation(content, speaker, turn)`** — rates importance with `gpt-4o-mini`, embeds with `text-embedding-3-small`, appends, decrements `importance_trigger`.
- **`should_reflect()` / `reflect(current_turn)`** — when `importance_trigger` (init 18) hits 0, pulls the N=10 most recent non-reflection nodes, calls `gpt-4o` for 2 first-person insights, embeds + appends them with `source="reflection"` and `importance=8`, resets the trigger. Ports `cognitive_modules/reflect.py`, simplified (no focal-question step, no reflection-of-reflections).

### Simulation loop (`simulation.py`)
Simplified Perceive → Plan → Converse loop from Park et al. (2023):
1. `load_agent(path)` — deserializes an agent JSON into an `Agent` object
2. Each agent is paired with its own `AssociativeMemory`, seeded from `bootstrap_memories`.
3. On each turn, for the speaking agent:
   - If `memory.should_reflect()`, call `memory.reflect()` **before** retrieval so new insights are eligible for top-k scoring.
   - `build_retrieval_query(agent, history)` → `agent.currently` + last two utterances.
   - `memory.retrieve(query, turn, k=5)` → top-k nodes.
   - `build_system_prompt(agent, scene_context, retrieved_memories)` — ISS + retrieved memories + constraints.
   - `agent_turn(...)` calls `gpt-4o` via `client.chat.completions.create` and returns the utterance.
4. After the utterance, every *other* agent writes it into their own stream via `add_observation`. The speaker does not self-observe.
5. `save_log(...)` writes `logs/run_<timestamp>.json` with `per_turn` (query + retrieved memories + reflection audit + utterance) and `final_memory_streams` for all agents.

Turn order is scripted (Rosa → Leonora → Rosa → Leonora → Max → Rosa → Max → Leonora → Rosa → Max), a deliberate deviation from Stanford Town's dynamic converse module — appropriate for a single-scene historical set-piece.

### Key interaction scene
The three agents converge on the 9th floor cutting room at 4:30 PM on March 25, 1911 — 30 minutes before the fire. Rosa (organizer, banned from the building) approaches Leonora (worker) about signing a union card; Max (supervisor) interrupts. The scene makes each agent's structural constraints explicit through their dialogue choices.

## Stanford Town reference
Upstream repo: <https://github.com/joonspk-research/generative_agents> (Park et al., UIST '23, arXiv:2304.03442). The adjacent `../Stanford_Town/repo/` contains the full implementation (Django frontend + Reverie backend) for reference. Key files:
- Cognitive loop: `reverie/backend_server/persona/persona.py`
- Conversation generation: `reverie/backend_server/persona/cognitive_modules/converse.py`
- Associative memory: `reverie/backend_server/persona/memory_structures/associative_memory.py`
- Retrieval scoring: `reverie/backend_server/persona/cognitive_modules/retrieve.py`
- Reflection: `reverie/backend_server/persona/cognitive_modules/reflect.py`
- ISS string: `reverie/backend_server/persona/memory_structures/scratch.py:382-414`
- Agent identity format: `environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/personas/Isabella Rodriguez/bootstrap_memory/scratch.json`

Stanford Town uses OpenAI `text-davinci-003` (deprecated via the legacy Completions API); this project uses the modern OpenAI Chat Completions API with `gpt-4o` (agent turns + reflection insights), `gpt-4o-mini` (importance scoring), and `text-embedding-3-small` (memory embeddings).

What we **did not** port: `perceive.py`, `plan.py`, `execute.py`, the maze / A* pathfinding, the Django/Phaser frontend, and hourly schedule generation — all scoped for a 25-agent open-world simulation running multiple in-game days, none of which is needed for a single fixed-scene 3-agent conversation.
