# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDUC_5919 course project implementing a historical generative agent simulation of the Triangle Shirtwaist Factory (NYC, March 25, 1911). Three LLM-powered agents — a garment worker, a factory supervisor/partial owner, and a union organizer — interact in a historically constrained scene to make systemic labor dynamics visible to learners.

See `simulation_plan.md` for the full design: historical context, agent identities, learning objective, and key interaction.

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

### Run the Jupyter notebook version
```bash
jupyter notebook simulation.ipynb
```

## Architecture

### Agent data (`agents/*.json`)
Each agent file mirrors the Stanford Town `scratch.json` pattern with fields:
- `innate` / `learned` / `currently` — layered identity (permanent traits → stable background → current goal)
- `bootstrap_memories` — list of text memories seeded at simulation start (what the agent knows, has seen, fears)
- `constraints` — explicit behavioral constraints passed into the system prompt to keep responses historically grounded
- `daily_plan_req` — the agent's standing purpose used for planning prompts

### Simulation loop (`simulation.py`)
Simplified Perceive → Plan → Converse loop from Park et al. (2023):
1. `load_agent(path)` — deserializes an agent JSON into an `Agent` object
2. `build_system_prompt(agent, scene_context)` — assembles the OpenAI system prompt from agent identity + historical scene description
3. `agent_turn(agent, conversation_history, client)` — calls `gpt-4o` via `client.chat.completions.create` and returns the agent's next speech/action
4. `run_interaction(agents, scene, turns)` — orchestrates the multi-turn exchange and prints the transcript

Each agent's turn is a separate OpenAI API call; the full conversation history is passed as context so agents respond to one another.

### Key interaction scene
The three agents converge on the 9th floor cutting room at 4:30 PM on March 25, 1911 — 30 minutes before the fire. Rosa (organizer, banned from the building) approaches Leonora (worker) about signing a union card; Max (supervisor) interrupts. The scene makes each agent's structural constraints explicit through their dialogue choices.

## Stanford Town reference
The adjacent `../Stanford_Town/repo/` contains Park et al.'s full implementation (Django frontend + Reverie backend) for reference. Key files:
- Cognitive loop: `reverie/backend_server/persona/persona.py`
- Conversation generation: `reverie/backend_server/persona/cognitive_modules/converse.py`
- Agent identity format: `environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/personas/Isabella Rodriguez/bootstrap_memory/scratch.json`

Stanford Town uses OpenAI `text-davinci-003` (deprecated via the legacy Completions API); this project uses the modern OpenAI Chat Completions API with `gpt-4o`.
