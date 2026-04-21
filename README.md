# Triangle Shirtwaist Factory — Historical Generative Agent Simulation

**EDUC_5919 — Deep Learning and Transformer Models · Spring 2026 · John Baker**

A three-agent generative simulation of the Triangle Shirtwaist Factory on the afternoon of **March 25, 1911** — thirty minutes before the fire that killed 146 garment workers in lower Manhattan. The goal is not to re-create the fire. It is to make visible the structural forces — economic desperation, identity and self-interest, legal permissiveness — that produced the catastrophe through many locally "rational" choices.

The architecture adapts **Stanford Town** (Park et al., 2023) to the modern OpenAI API. Three LLM-powered agents converse in a fixed scene; each agent's turn is one `chat.completions` call with the agent's identity, bootstrap memories, and behavioral constraints assembled into a system prompt.

This repository is the complete submission for the EDUC_5919 course project and is published for classmates to read, reuse, and adapt for their own historical worlds.

---

## Table of contents

1. [What's in this repo](#whats-in-this-repo)
2. [Quick start](#quick-start)
3. [The three agents](#the-three-agents)
4. [Architecture](#architecture)
5. [How to adapt this for your own historical world](#how-to-adapt-this-for-your-own-historical-world)
6. [What was actually built, step by step](#what-was-actually-built-step-by-step)
7. [Reading the report and the logs](#reading-the-report-and-the-logs)
8. [Stanford Town reference map](#stanford-town-reference-map)
9. [Known limitations](#known-limitations)
10. [Credits and references](#credits-and-references)

---

## What's in this repo

```
.
├── README.md                  — this file
├── assignment.md              — the EDUC_5919 assignment brief
├── simulation_plan.md         — the design doc (world, agents, scene, learning objective)
├── CLAUDE.md                  — guidance for working on this repo with Claude Code
│
├── requirements.txt           — openai, jupyter
├── simulation.py              — CLI entry point
├── simulation.ipynb           — Jupyter notebook version (same logic, narrated)
│
├── agents/
│   ├── leonora_russo.json     — garment worker, 22, Sicilian immigrant
│   ├── max_bernstein.json     — floor supervisor / partial owner, 48, from Minsk
│   └── rosa_peretz.json       — ILGWU organizer, 24, survivor of 1909 Uprising
│
├── logs/
│   └── run_YYYYMMDD_HHMMSS.json   — one JSON log per simulation run
│
├── report.md                  — the assignment report, in markdown
└── report.pdf                 — the assignment report, in PDF (the submission artifact)
```

---

## Quick start

### 1. Clone and set up the environment

```bash
git clone https://github.com/baker-jr-john/educ5919-triangle-simulation.git
cd educ5919-triangle-simulation

python3 -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get an OpenAI API key

Create one at <https://platform.openai.com/api-keys>. The default model is `gpt-4o`; a full 10-turn run costs roughly $0.05 USD.

### 3. Run the simulation

```bash
export OPENAI_API_KEY="sk-..."
python simulation.py
```

You'll see the transcript stream to stdout, and a new JSON log will be written to `logs/run_<timestamp>.json`.

### 4. Run the notebook version

```bash
jupyter notebook simulation.ipynb
```

The notebook has the same logic as `simulation.py`, split across narrated cells. "Run All" produces an equivalent transcript inline.

---

## The three agents

Each agent JSON mirrors Stanford Town's `scratch.json` layering — `innate` / `learned` / `currently` (permanent traits → stable background → present goal) — plus a flat `bootstrap_memories` list and an explicit `constraints` list.

### Leonora Russo (22, garment worker)
- Sicilian, three years in New York, main wage-earner since her father's accident
- Wants: keep her job, save for her sister Concetta's passage from Palermo
- Constrained by: family debt, the informal blacklist, immigration precarity

### Max Bernstein (48, floor supervisor and partial owner)
- Jewish immigrant from Minsk, worked up from cutter to owner over twenty years
- Wants: meet output quota, prevent theft, keep organizers out
- Constrained by: pressure from Harris & Blanck, his own self-made-man identity

### Rosa Peretz (24, ILGWU organizer)
- From Białystok via pogrom, veteran of the 1909 Uprising of the 20,000
- Wants: get one union card signed (Leonora's, named at an Elizabeth Street meeting)
- Constrained by: banned from the building, a third arrest carries a heavy sentence

---

## Architecture

### Turn loop

```
for agent in scripted turn order:
    system_prompt = (
        scene_context
        + "You are <Name>." + ISS(agent)
        + "What is in your head:" + bootstrap_memories
        + "Constraints on your behavior:" + constraints
        + output instructions
    )
    user_message = "Conversation so far: ..." + "What does <Name> say next?"
    utterance = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", ...}, {"role": "user", ...}],
    )
    history.append((agent.first_name, utterance))
```

### Key functions in `simulation.py`

| Function | Purpose |
|----------|---------|
| `load_agent(path)` | Deserialize an agent JSON into an `Agent` dataclass |
| `get_str_iss(agent)` | Build the Identity Stable Set string — direct port of Stanford Town's `Scratch.get_str_iss()` |
| `build_system_prompt(agent, scene_context)` | Assemble the full system prompt for one turn |
| `agent_turn(agent, history, client)` | One turn = one OpenAI API call; return the utterance |
| `run_interaction(agents, turn_order, client)` | Orchestrate the scripted turn sequence; print transcript |
| `save_log(history, agents, turn_order)` | Write `logs/run_<timestamp>.json` |

### Prompt architecture — why it's built this way

The three key Stanford Town patterns we reuse are:

1. **Identity stable set (ISS).** A single string containing `Name / Age / Innate traits / Learned traits / Currently / Daily plan / Current date`. See `simulation.py:get_str_iss()` and Stanford Town's `reverie/backend_server/persona/memory_structures/scratch.py:382-414`.

2. **Iterative conversation prompt structure.** Persona ISS → retrieved memory → past context → current location → current context → conversation so far → "what should X say next?". See Stanford Town's `reverie/backend_server/persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt`.

3. **Agent JSON shape.** Mirrors Stanford Town's `scratch.json` but restricted to identity fields (no pathfinding, no maze state, no hourly scheduling).

What we **dropped** from Stanford Town:

- Embedding-based memory retrieval (`AssociativeMemory`)
- The Django frontend / Phaser 3 game world
- The `perceive` → `retrieve` → `plan` → `reflect` → `execute` pipeline
- Hourly schedule generation and action decomposition
- The maze / A* pathfinding

All of that infrastructure exists in Stanford Town to support a 25-agent open-world simulation running for several in-game days. For a single-scene 3-agent conversation, most of it is unnecessary — bootstrap memories go directly into the system prompt, and the turn order is scripted.

---

## How to adapt this for your own historical world

This is the section most useful to classmates doing the same assignment with a different world. The short version: **write the content, keep the code.**

### Step 1. Pick your world (a scene, not an era)

The tightest simulations pick a narrow moment where three people with different structural positions are forced into the same room. Good candidates:

- A Qing dynasty market town on the day of a tax collector's visit
- A Depression-era Oklahoma tenant-farm eviction
- A Memphis sanitation worker, a white city council member, and Martin Luther King Jr.'s advance man, February 1968
- A freed slave, a Union officer, and a former plantation overseer, at a Freedmen's Bureau office, 1867

Pick one where the story is *usually* told as "good person vs. bad person" but actually requires structural explanation. That's where the simulation will earn its keep.

### Step 2. Write the three agent JSONs

Copy `agents/leonora_russo.json` as a template. The fields are:

| Field | Purpose |
|-------|---------|
| `name`, `first_name`, `last_name`, `age` | Basic identity |
| `innate` | A short list of permanent traits, comma-separated |
| `learned` | 1–2 sentences of stable background |
| `currently` | 1–3 sentences describing what the agent is doing and wanting *on this specific day* |
| `daily_plan_req` | A single sentence naming the agent's standing purpose |
| `bootstrap_memories` | List of 6–10 first-person memories grounded in the historical record |
| `constraints` | List of explicit behavioral rules, including "no anachronisms, no modern references" |

**The memories do the most work.** Every memory should be something an agent might *invoke in argument*. Generic memories ("I grew up poor") produce generic dialogue. Specific memories ("My cousin Giulia was beaten in the street by men the bosses hired in December 1909") produce specific, period-grounded arguments.

**The constraints are the second-most-important field.** Without explicit rules about period voice, the model drifts into modern American English. Without explicit rules about turn length, turns balloon to 8–10 sentences. Example constraints that worked:

```
"Stay in period voice. No references to OSHA, the 40-hour week, modern labor law,
 the later history of the Triangle fire, or anything after March 1911.",
"Keep your turn SHORT — one to three sentences, or a brief action in asterisks.",
"You DO NOT know the fire will happen in thirty minutes."
```

### Step 3. Write the scene context

Edit `SCENE_CONTEXT` at the top of `simulation.py`. This is a ~200-word module-level string describing:

- The date and time, exactly
- The physical setting (what's in the room, what's audible, what's visible)
- Who is doing what when the scene opens
- What **none of the agents know** (this is crucial — it prevents foreshadowing)

### Step 4. Write the turn order

Edit the `turn_order` list in `main()`. Scripted turn-taking is appropriate for a single-scene set-piece; you do not need Stanford Town's dynamic `converse` module. Choose an order that gives each agent enough turns to demonstrate their constraints.

### Step 5. Run a few passes, collect logs, analyze

```bash
for i in 1 2 3; do python simulation.py; done
```

Each run writes a `logs/run_<timestamp>.json`. Read the transcripts looking for:

- **Behaviors the constraints specified** (did the agent stay in period voice?)
- **Behaviors the constraints produced but did not specify** (these are your best observations — the emergent ones)
- **Anachronisms or historical failures** (these become your plausibility evaluation)
- **Patterns that repeat across runs** (these show the structural forces doing their work)

### Step 6. Write the report

See `report.md` for the structure used here. The eight sections from `assignment.md` each get their own heading; the Behavior Documentation section cites specific lines from specific JSON logs (`run_<ts>.json / turn N`) as evidence.

---

## What was actually built, step by step

For classmates who want to see the whole arc, not just the final artifact, here is what the development process looked like:

### Phase 1 — Understand Stanford Town (read, don't run)

We did **not** successfully run Stanford Town end-to-end. The code targets the deprecated OpenAI `text-davinci-003` model and has CUDA/Python/package-version dependencies that are painful to reproduce in 2026. Instead we:

- Read `reverie/backend_server/persona/memory_structures/scratch.py` to understand the identity fields
- Read `reverie/backend_server/persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt` to understand the conversation prompt structure
- Read a sample `scratch.json` (for Isabella Rodriguez in the `base_the_ville_isabella_maria_klaus` base simulation) to see the JSON shape

That was the full dependency on Stanford Town — reading three files.

### Phase 2 — Write the plan

`simulation_plan.md` was written before any code. It fixes:
- The historical context (Triangle fire, why it requires structural framing)
- The three agent identities (what each wants, what constrains each)
- The learning objective (what students should understand afterward)
- The key interaction scene (9th floor, 4:30 PM, Rosa approaches Leonora, Max intervenes)

**Doing this in prose first, before touching code, was the single most useful step in the project.** If the scene is not clear in prose, no amount of good prompt engineering will rescue it.

### Phase 3 — Build the simulation

1. Write `requirements.txt` (originally `anthropic`, later swapped to `openai`)
2. Write three agent JSONs (Leonora, Max, Rosa) with 9 memories and 6–8 constraints each
3. Write `simulation.py` with the 5 functions in the architecture table above
4. Write `simulation.ipynb` as a self-contained notebook version with narration between cells
5. Smoke-test: confirm agents load, prompts assemble, ISS strings render correctly — before making any API call

### Phase 4 — Run and log

The live run was first attempted with Claude (claude-sonnet-4-6) but swapped to OpenAI (gpt-4o) to reuse existing course credits. The swap touched three files: `requirements.txt`, `simulation.py`, `simulation.ipynb` — about 20 lines of changes. No agent JSONs changed. This is one of the virtues of the layered design: the model swap is a plumbing change.

Three runs were executed, producing the three JSON logs in `logs/`.

### Phase 5 — Analyze and write the report

Five specific behavior observations were extracted from the logs — each cited to a run and turn number. Patterns across runs were identified (power revealed through action, Leonora defers rather than refuses, Rosa's specificity holds under interruption). The report's most important argument, in §6, is about the **limits** of the simulation as a learning tool.

### Phase 6 — Convert to PDF

`report.md` is the authoritative source. `report.pdf` was produced via `pandoc report.md -o report.pdf --pdf-engine=weasyprint`. Either file can be regenerated; the PDF is the submission artifact.

---

## Reading the report and the logs

### The report (`report.pdf`, `report.md`)

14 pages, eight sections that map 1-to-1 to the assignment's required components. Section 6 ("Simulation, Context, and Critique") is the one I most encourage classmates to read, because it is the hardest part of the assignment to do well — it's where you have to explain why the thing you just built is not sufficient on its own to teach history.

### The JSON logs (`logs/run_*.json`)

Each log contains:

```json
{
  "timestamp": "20260420_195823",
  "model": "gpt-4o",
  "scene_context": "It is Saturday, March 25, 1911...",
  "turn_order": ["rosa", "leonora", "rosa", ...],
  "agents": { "leonora": {...}, "max": {...}, "rosa": {...} },
  "transcript": [
    {"speaker": "Rosa", "utterance": "..."},
    {"speaker": "Leonora", "utterance": "..."},
    ...
  ]
}
```

You can rerun any turn offline by loading the agent JSONs, replaying the history up to turn N, and calling `agent_turn()` on the N+1 agent. No state is hidden.

---

## Stanford Town reference map

For classmates who want to trace back to the source, these are the three files from Stanford Town that this project actually depends on (read, not run):

- **Agent identity format:** `Stanford_Town/repo/environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/personas/Isabella Rodriguez/bootstrap_memory/scratch.json`
- **ISS string implementation:** `Stanford_Town/repo/reverie/backend_server/persona/memory_structures/scratch.py:382-414` (the `get_str_iss()` method)
- **Conversation prompt template:** `Stanford_Town/repo/reverie/backend_server/persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt`

If you want to see the full cognitive loop that we **did not** port, start at `Stanford_Town/repo/reverie/backend_server/persona/persona.py` and follow `Persona.move()` through `perceive.py` → `retrieve.py` → `plan.py` → `execute.py` → `converse.py` → `reflect.py`. Impressive, but overkill for a single-scene conversation.

---

## Known limitations

These are the same limitations documented in `report.md` §5 and §7. They are worth restating up front for anyone thinking about reusing this code:

- **Language is over-fluent.** Even when constraints specify "broken English," the model's output is fluent speech performing brokenness. This is a well-known LLM failure mode.
- **The scene is over-legible.** Real factory-floor conversation in 1911 would have been interrupted, noisy, cut off mid-sentence. The simulation gives each agent a clean turn.
- **Composition is fine, content is generated.** Nothing any agent says is attested. They are plausible composite characters, not historical subjects. Do not let students quote the transcripts as if they were primary sources.
- **The interpretation is baked in.** By writing the agents' memories and constraints the way I did, I committed the simulation to a structural reading of the fire. That reading is defensible, but it is a reading — not a neutral observation.

The simulation is a **rehearsal of structural reasoning**, not a **source of historical content**. It must be scaffolded before and critiqued after. See `report.md` §6 for how.

---

## Credits and references

**Course:** EDUC_5919 — Deep Learning and Transformer Models, Spring 2026

**Stanford Town (Generative Agents):**
Park, Joon Sung, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. "Generative Agents: Interactive Simulacra of Human Behavior." *UIST '23.* arXiv:2304.03442.
Code: <https://github.com/joonspk-research/generative_agents>

**Historiography of the Triangle fire:**
- Stein, Leon. *The Triangle Fire.* Lippincott, 1962.
- Von Drehle, David. *Triangle: The Fire That Changed America.* Grove Press, 2003.
- Orleck, Annelise. *Common Sense and a Little Fire: Women and Working-Class Politics in the United States, 1900–1965.* University of North Carolina Press, 1995.

**Built with:** OpenAI `gpt-4o`, Python 3.9+, Jupyter, pandoc + weasyprint.

**Licensed for classroom reuse.** Adapt the agents, adapt the scene, adapt the report structure — the code is the boring part; the historical thinking is the point.
