# Triangle Shirtwaist Factory — Historical Generative Agent Simulation

**EDUC_5919 — Deep Learning and Transformer Models · Spring 2026 · John Baker**

A three-agent generative simulation of the Triangle Shirtwaist Factory on the afternoon of **March 25, 1911** — thirty minutes before the fire that killed 146 garment workers in lower Manhattan. The goal is not to re-create the fire. It is to make visible the structural forces — economic desperation, identity and self-interest, legal permissiveness — that produced the catastrophe through many locally "rational" choices.

The architecture adapts **Stanford Town** — Park et al.'s *Generative Agents: Interactive Simulacra of Human Behavior* ([joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents), [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)) — to the modern OpenAI API. Three LLM-powered agents converse in a fixed scene; each agent's turn is one `chat.completions` call with the agent's identity, retrieved memories, and behavioral constraints assembled into a system prompt. The cognitive modules ported from Stanford Town are **associative memory with weighted retrieval** and **importance-triggered reflection**; everything else (hourly scheduling, pathfinding, the Phaser game world) is out of scope for a single-scene three-agent conversation.

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
├── LICENSE                    — MIT license
├── .gitignore                 — files git should not track (venv, caches, secrets, scratch)
├── assignment.md              — the EDUC_5919 assignment brief
├── simulation_plan.md         — the design doc (world, agents, scene, learning objective)
├── CLAUDE.md                  — guidance for working on this repo with Claude Code
│
├── requirements.txt           — openai, jupyter
├── simulation.py              — CLI entry point (turn loop + scene + logging)
├── memory.py                  — AssociativeMemory: retrieval (recency + relevance + importance) + reflection
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

Create one at <https://platform.openai.com/api-keys>. The default models are `gpt-4o` for agent turns and reflection insights, `text-embedding-3-small` for memory embeddings, and `gpt-4o-mini` for scoring observation importance. A full 10-turn run costs roughly $0.15 USD.

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
seed each agent's AssociativeMemory with their bootstrap_memories (all embedded)

for turn_num, agent in enumerate(scripted turn order):
    # 0. Reflect first, so fresh insights are eligible for retrieval this turn.
    if agent.memory.should_reflect():
        insights = agent.memory.reflect(current_turn=turn_num)
        # each insight is embedded and appended to the stream with
        # source="reflection" and elevated importance

    query = agent.currently + " " + (last two utterances heard)
    retrieved = agent.memory.retrieve(query, current_turn=turn_num, k=5)

    system_prompt = (
        scene_context
        + "You are <Name>." + ISS(agent)
        + "What is in your head right now:" + retrieved         # <-- top-k only, not all bootstrap
        + "Constraints on your behavior:" + constraints
        + output instructions
    )
    user_message = "Conversation so far: ..." + "What does <Name> say next?"
    utterance = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", ...}, {"role": "user", ...}],
    )
    history.append((agent.first_name, utterance))

    # every other agent hears this and writes it into their own memory,
    # with an importance score rated by gpt-4o-mini. add_observation()
    # decrements that agent's importance_trigger by the rated importance;
    # when it hits 0, should_reflect() returns true on their next turn.
    for other in other_agents:
        other.memory.add_observation(f'{agent.name} said: "{utterance}"', ...)
```

### Memory and retrieval (`memory.py`)

Each agent owns an `AssociativeMemory`. Nodes come from two sources:

- **Bootstrap memories** — seeded at load, embedded with `text-embedding-3-small`, given a default importance of 6/10.
- **Observations** — every utterance the other two agents produce; embedded at write time; importance rated 1–10 by a `gpt-4o-mini` call. The speaker does not self-observe (the conversation history already covers that).

On each turn, before the `gpt-4o` call, the speaking agent's memory is scored against a retrieval query. Following Park et al. (2023) §A.1.1:

```
score(node) = W_RECENCY   * recency_norm(node)
            + W_RELEVANCE * relevance_norm(node)
            + W_IMPORTANCE * importance_norm(node)

recency    = 0.995 ^ (current_turn - last_accessed_turn)
relevance  = cosine(embed(query), node.embedding)
importance = node.importance / 10
```

Each of the three components is min-max normalized across the candidate set before weighting. Defaults mirror Stanford Town's `scratch.json` (all three weights = 1.0, recency decay = 0.995). The top-k nodes (default k=5) have their `last_accessed_turn` bumped — so memories that get retrieved frequently decay more slowly.

### Reflection (`memory.py`)

Observation nodes give an agent access to *what was said*. Reflection nodes give it access to *what that means*.

Each `AssociativeMemory` tracks an `importance_trigger` counter initialized to `REFLECTION_THRESHOLD = 18`. Every new observation decrements the trigger by its rated importance (1–10). When the trigger hits 0, `should_reflect()` returns true, and at the top of that agent's next turn — *before* retrieval runs — `reflect()` fires:

1. Pull the N most recent non-reflection nodes from the stream (default N=10).
2. Ask `gpt-4o` for `REFLECTION_N_INSIGHTS = 2` first-person sentences ("I see that…", "I am beginning to realize…") drawn from those memories, in the agent's own voice and period.
3. Embed each insight, append it as a new node with `source="reflection"` and `importance=8`, and reset the trigger.

Because reflection fires *before* retrieval for the same turn, fresh insights are candidates for top-k scoring and can appear in the system prompt immediately. Because they live in the same stream as bootstrap + observations, they also remain available for later turns — so a reflection Rosa makes at turn 4 ("this man believes what he is saying, which means argument will not reach him") can be surfaced at turn 8 when she's deciding how to close the scene.

Bootstrap nodes do **not** decrement the trigger; they represent prior life, not fresh experience. Reflection nodes are excluded from their own input pool to keep insights grounded in observations rather than recursively rewriting themselves.

Threshold tuning is scene-dependent. Stanford Town uses 150 accumulated importance for multi-day runs; we use 18 so each agent reflects roughly once or twice in a ten-turn scene. Lower the threshold to see reflection more often; raise it (or set it impossibly high) to turn reflection off.

### Key functions

| File / Function | Purpose |
|---|---|
| `simulation.py :: load_agent(path)` | Deserialize an agent JSON into an `Agent` dataclass |
| `simulation.py :: get_str_iss(agent)` | Identity Stable Set string — port of Stanford Town's `Scratch.get_str_iss()` |
| `simulation.py :: build_retrieval_query(agent, history)` | `agent.currently + last two utterances` → one query string |
| `simulation.py :: build_system_prompt(agent, scene, retrieved_memories)` | Assemble the full per-turn system prompt |
| `simulation.py :: agent_turn(agent, memory, history, turn_num, client)` | Retrieve → `chat.completions` → return `(utterance, retrieved_strs)` |
| `simulation.py :: run_interaction(agents, memories, turn_order, client)` | Orchestrate turns; after each utterance, record observations in every other agent's memory |
| `simulation.py :: save_log(...)` | Write `logs/run_<timestamp>.json` including per-turn retrieval audit and final memory streams |
| `memory.py :: AssociativeMemory.seed_bootstrap(list)` | Embed and insert each bootstrap memory as a node |
| `memory.py :: AssociativeMemory.add_observation(content, speaker, turn)` | Rate importance, embed, append an observation node, decrement `importance_trigger` |
| `memory.py :: AssociativeMemory.should_reflect()` | True when `importance_trigger` has crossed zero |
| `memory.py :: AssociativeMemory.reflect(current_turn)` | Pull recent N memories, generate first-person insights with `gpt-4o`, embed + append them as `source="reflection"` nodes, reset trigger |
| `memory.py :: AssociativeMemory.retrieve(query, current_turn, k)` | Score + sort + return top-k; bump `last_accessed_turn` |
| `memory.py :: rate_importance(content, agent)` | One `gpt-4o-mini` call; return an integer 1–10 |
| `memory.py :: generate_reflection_insights(agent, recent)` | One `gpt-4o` call; return first-person insight strings |

### Prompt architecture — why it's built this way

The five key Stanford Town patterns we reuse are:

1. **Identity stable set (ISS).** A single string containing `Name / Age / Innate traits / Learned traits / Currently / Daily plan / Current date`. See `simulation.py:get_str_iss()` and Stanford Town's `reverie/backend_server/persona/memory_structures/scratch.py:382-414`.

2. **Iterative conversation prompt structure.** Persona ISS → retrieved memory → past context → current location → current context → conversation so far → "what should X say next?". See Stanford Town's `reverie/backend_server/persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt`.

3. **Associative memory with weighted retrieval.** Nodes + embeddings + the `recency + relevance + importance` scoring rule from Park et al. (2023) §A.1.1. See `memory.py` and Stanford Town's `reverie/backend_server/persona/memory_structures/associative_memory.py` + `reverie/backend_server/persona/cognitive_modules/retrieve.py`.

4. **Importance-triggered reflection.** Accumulated observation-importance crosses a threshold → the agent generates high-level first-person insights from recent memories, writes them back into the stream as new retrievable nodes. See `memory.py:reflect()` and Stanford Town's `reverie/backend_server/persona/cognitive_modules/reflect.py`.

5. **Agent JSON shape.** Mirrors Stanford Town's `scratch.json` but restricted to identity fields (no pathfinding, no maze state, no hourly scheduling).

What we **simplified** relative to Stanford Town:

- **Single-field nodes.** Stanford Town splits each node into `(subject, predicate, object) + description`; our nodes are single `content` strings. The decomposition is used by their reflect module to build a thought tree; we don't go that deep — reflection writes free-form first-person sentences.
- **Embeddings only, no keyword fallback.** Stanford Town also scores by keyword-strength overlap as a secondary signal. We use embeddings only.
- **Single retrieval query.** Stanford Town's `new_retrieve` takes a list of focal points and unions the results; we concatenate `currently + last two utterances` into one query.
- **Reflection skips the focal-question step.** Stanford Town first generates 3–5 focal questions from recent memories, then retrieves per question, then synthesizes insights. We go directly from most-recent-N to insights, because for a ten-turn scene the question-generation step is overhead.
- **No reflection-of-reflections.** Stanford Town lets reflection nodes re-enter the reflection pool, producing a thought hierarchy. We exclude them to keep insights grounded in observations.
- **No self-observation.** The conversation history already contains each agent's own utterances, so the speaker does not add a node when they finish a turn.

What we **dropped** entirely:

- The Django frontend / Phaser 3 game world
- Hourly schedule generation and action decomposition (`plan.py`)
- `perceive.py` / `execute.py` / spatial `converse.py` (replaced by scripted turn order)
- The maze / A* pathfinding

All of that infrastructure exists in Stanford Town to support a 25-agent open-world simulation running for several in-game days. For a single-scene 3-agent conversation, most of it is unnecessary — scripted turn order, one room, ten turns. What Stanford Town's *cognitive* modules add (retrieval + reflection) is exactly what makes the conversation non-trivial, which is why we ported those and nothing else.

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

Also skim the `per_turn` retrieval audit: did the retrieval module pull memories that actually fit the moment, or did it anchor on bootstrap memories that an agent would have no reason to be thinking about right now? If retrieval looks wrong, the most likely causes are (a) bootstrap memories are too generic to differentiate by cosine distance, or (b) the retrieval query is too narrow; widen it by including more of the recent history.

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

### Phase 3 — Build the simulation (first pass)

1. Write `requirements.txt` (originally `anthropic`, later swapped to `openai`)
2. Write three agent JSONs (Leonora, Max, Rosa) with 9 memories and 6–8 constraints each
3. Write `simulation.py` with the core turn functions
4. Write `simulation.ipynb` as a self-contained notebook version with narration between cells
5. Smoke-test: confirm agents load, prompts assemble, ISS strings render correctly — before making any API call

### Phase 4 — Run and log

The live run was first attempted with Claude (claude-sonnet-4-6) but swapped to OpenAI (gpt-4o) to reuse existing course credits. The swap touched three files: `requirements.txt`, `simulation.py`, `simulation.ipynb` — about 20 lines of changes. No agent JSONs changed. This is one of the virtues of the layered design: the model swap is a plumbing change.

Three runs were executed, producing the initial JSON logs in `logs/`.

### Phase 5a — Retrofit: add associative memory with retrieval

After the first submission to the professor, her feedback was that the simulation was **missing infrastructure from Stanford Town** — specifically, the memory stream and retrieval module. The first-pass version stuffed every bootstrap memory into every prompt, which is not how Park et al.'s generative agents actually work.

This phase addressed that by adding `memory.py`:

1. Port `AssociativeMemory` from Stanford Town's `memory_structures/associative_memory.py`, with single-field `content` nodes (no `(subject, predicate, object)` decomposition — unnecessary without reflection).
2. Port the retrieval scoring from `cognitive_modules/retrieve.py`: min-max normalized `recency + relevance + importance`, weights all 1.0, recency decay 0.995.
3. Embed every memory node with `text-embedding-3-small` (a cheap, modern replacement for Stanford Town's OpenAI ada embeddings).
4. For every observation written into a memory stream, rate its importance 1–10 with a `gpt-4o-mini` call — the "poignancy score" from Park et al. §A.1.1.
5. Rewire `simulation.py` and `simulation.ipynb` so that each turn first retrieves the top-k memories for the current speaker, injects *only those* into the system prompt, and then writes the resulting utterance into every other agent's memory stream as a new observation.
6. Extend `save_log()` to include a `per_turn` retrieval audit (query + top-k memory contents) and each agent's `final_memory_streams`, so graders can see what was retrieved when.

The reasoning for choosing faithful retrieval over a cheaper keyword-overlap fallback: faithfulness to Stanford Town *was the point* of the retrofit.

### Phase 5b — Add reflection

After retrieval was working, the next piece of Stanford Town's cognitive loop worth porting was **reflection** — the mechanism by which agents synthesize recent memories into higher-level first-person insights. Without reflection, an agent has access to *what was said* but not to *what it meant*; every turn reasons over raw events only.

The addition layered cleanly on top of the retrieval module. In `memory.py`:

1. Each `AssociativeMemory` gained an `importance_trigger` counter. Every observation decrements it by its rated importance; when it hits zero, `should_reflect()` returns true.
2. `reflect()` pulls the N most recent non-reflection nodes, calls `gpt-4o` with a prompt that asks for two first-person insights ("I see that…", "I am beginning to realize…") in the agent's voice and period, embeds each insight, and writes it back into the stream with `source="reflection"` and elevated importance.
3. In `simulation.py`, reflection fires at the top of each turn *before* retrieval, so fresh insights are eligible to be pulled into that same turn's system prompt.

Threshold chosen so each agent reflects roughly once or twice in a ten-turn scene (18, vs. Stanford Town's 150 for multi-day runs).

Planning (the other major piece of Stanford Town's cognitive loop) was considered and intentionally skipped — it targets hourly schedules across in-game days, which the `turn_order` list already handles for a scripted single-scene conversation.

### Phase 6 — Analyze and write the report

Five specific behavior observations were extracted from the logs — each cited to a run and turn number. Patterns across runs were identified (power revealed through action, Leonora defers rather than refuses, Rosa's specificity holds under interruption). The report's most important argument, in §6, is about the **limits** of the simulation as a learning tool.

### Phase 7 — Convert to PDF

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
  "retrieval_k": 5,
  "scene_context": "It is Saturday, March 25, 1911...",
  "turn_order": ["rosa", "leonora", "rosa", ...],
  "agents": { "leonora": {...}, "max": {...}, "rosa": {...} },
  "per_turn": [
    {
      "turn": 0,
      "speaker": "Rosa",
      "reflection": null,
      "retrieval_query": "Get Leonora's signature on a union card...",
      "retrieved_memories": ["I was in Union Square...", "The fire escape...", ...],
      "utterance": "Leonora, keep cutting. Don't look up..."
    },
    {
      "turn": 5,
      "speaker": "Rosa",
      "reflection": {
        "triggered": true,
        "insights": [
          "I am beginning to realize that Max's certainty is not bluster — it is belief, and it will not bend to argument.",
          "It is becoming clear to me that Leonora hears me, but what she hears louder is the weight of her family."
        ]
      },
      "retrieval_query": "...",
      "retrieved_memories": [...],
      "utterance": "..."
    },
    ...
  ],
  "transcript": [
    {"speaker": "Rosa", "utterance": "..."},
    {"speaker": "Leonora", "utterance": "..."},
    ...
  ],
  "final_memory_streams": {
    "leonora": [
      { "content": "I came from Palermo...", "source": "bootstrap", "importance": 6, ... },
      { "content": "Rosa Peretz said: \"...\"", "source": "observation", "speaker": "Rosa Peretz", "importance": 7, ... },
      { "content": "I see that this woman will not give up, even with the supervisor here...", "source": "reflection", "speaker": "Leonora Russo", "importance": 8, ... },
      ...
    ],
    "max": [...],
    "rosa": [...]
  }
}
```

- **`per_turn.reflection`** is `null` on most turns, but when the importance trigger has fired it contains the two insight strings written back into that agent's memory stream *before* retrieval ran for the same turn. This is the reflection audit — it lets you see exactly what higher-level understanding each character drew from the confrontation, and when.
- **`per_turn.retrieval_query` + `retrieved_memories`** is the retrieval audit: for each turn, the query that was built and the top-k memory contents that were actually injected into that turn's system prompt.
- **`final_memory_streams`** is each agent's full post-scene memory: bootstrap seeds + observations + reflections, each with `source`, `speaker`, and importance. Embeddings are elided as `"[1536-dim vector]"` to keep the file human-readable. Filter for `"source": "reflection"` in any stream to see the insights that agent generated.

You can rerun any turn offline by loading the agent JSONs, replaying the history up to turn N, and calling `agent_turn()` on the N+1 agent. No state is hidden.

---

## Stanford Town reference map

**Upstream repository:** <https://github.com/joonspk-research/generative_agents> — Park, Joon Sung, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein, *Generative Agents: Interactive Simulacra of Human Behavior*, UIST '23. All of this project's cognitive-module work is a port from that code.

For classmates who want to trace back to the source, these are the files from Stanford Town that this project actually depends on (read, not run):

- **Agent identity format:** `Stanford_Town/repo/environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/personas/Isabella Rodriguez/bootstrap_memory/scratch.json`
- **ISS string implementation:** `Stanford_Town/repo/reverie/backend_server/persona/memory_structures/scratch.py:382-414` (the `get_str_iss()` method)
- **Conversation prompt template:** `Stanford_Town/repo/reverie/backend_server/persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt`
- **Associative memory class:** `Stanford_Town/repo/reverie/backend_server/persona/memory_structures/associative_memory.py` (node + stream + add-observation structure)
- **Retrieval scoring:** `Stanford_Town/repo/reverie/backend_server/persona/cognitive_modules/retrieve.py` (the `recency + relevance + importance` weighted sum)
- **Reflection:** `Stanford_Town/repo/reverie/backend_server/persona/cognitive_modules/reflect.py` (importance-trigger gating, focal-question generation, insight writing)
- **Poignancy rating prompt:** `Stanford_Town/repo/reverie/backend_server/persona/prompt_template/run_gpt_prompt.py :: generate_poig_score()`

If you want to see the rest of the cognitive loop that we **did not** port, start at `Stanford_Town/repo/reverie/backend_server/persona/persona.py` and follow `Persona.move()` through `perceive.py` → `plan.py` → `execute.py` → `converse.py`. We port `retrieve` and `reflect` but skip `perceive` / `plan` / `execute` — impressive, but overkill for a single-scene conversation.

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
