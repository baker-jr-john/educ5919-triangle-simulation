# Historical Generative Agent Simulation — Plan

## Context

Course project for EDUC_5919 (Deep Learning and Transformer Models). The assignment asks students to design and implement a 3-agent generative simulation of a historical context where understanding people's actions requires attention to social, economic, and cultural forces. The Stanford Town codebase (Park et al., 2023) is available as infrastructure reference; the Project directory is currently empty.

---

## 1. Historical Context: Triangle Shirtwaist Factory, NYC — March 25, 1911

The garment industry of lower Manhattan employed tens of thousands of immigrant women, mostly Italian and Jewish, under conditions of extreme economic vulnerability. Workers labored 14-hour days for poverty wages, with no job security, no safety protections, and no legal right to organize. Factory owners locked exit doors during shifts to prevent theft and unauthorized breaks. On March 25, 1911, a fire broke out at the Triangle Shirtwaist Company on the 8th–10th floors of the Asch Building. 146 workers died, most because the exits were locked or the fire escapes collapsed.

**Why this context requires social/economic/cultural framing:**
The fire is often taught as a simple story of negligence. It was not. Every actor made decisions that were "rational" within their constrained world: owners locked doors because labor laws permitted it and theft cost money; workers stayed at dangerous jobs because they had no alternative; organizers couldn't reach workers because employer retaliation was swift and legal. The deaths were produced by a system, not a villain. That requires context to see.

---

## 2. Three Agents

### Agent 1: Leonora Russo (garment worker, 22)
- **Who she is:** Arrived from Palermo, Sicily, three years ago with her father's debt to pay and a younger sister waiting to emigrate. Speaks broken English. Works the shirtwaist cutting table on the 9th floor.
- **What she wants:** Keep her job (her family depends on every dollar), earn enough to pay for her sister's passage to America
- **What constrains her:** Economic desperation she cannot escape; immigration status that makes confrontation dangerous; fear of the blacklist (owners share names of "troublemakers" across factories); trust only in people from her village

### Agent 2: Max Bernstein (factory floor supervisor / partial owner, 48)
- **Who he is:** Jewish immigrant who arrived from Minsk at 19, worked the floor himself, and over 20 years accumulated a small ownership stake in Triangle. Proud of what he built. Genuinely believes the factory gave him opportunity and can do the same for others who work hard.
- **What he wants:** Meet the weekly output quota, prevent the theft that eats his profit margin, keep labor organizers out of his factory
- **What constrains him:** Pressure from majority owners (Blanck & Harris) to cut costs; his own identity as a self-made man who distrusts collective action; fire codes he bends because compliance would cost money, and inspectors rarely visit

### Agent 3: Rosa Peretz (ILGWU labor organizer, 24)
- **Who she is:** Former shirtwaist worker who survived the 1909 Uprising of the 20,000 (the failed general strike). Now works for the International Ladies' Garment Workers' Union. Has seen the Triangle floor's locked exit doors and fears catastrophe.
- **What she wants:** Get workers to sign union cards, build collective power, prevent the disaster she sees coming
- **What constrains her:** Banned from the building by name; must work covertly; legal protection for organizing is limited; many workers are too afraid; the ILGWU itself is male-dominated and underestimates women workers

---

## 3. Learning Objective

**After engaging with this simulation, learners should be able to explain:**
> Why 146 people died on March 25, 1911 — not in terms of a single bad actor, but in terms of how economic desperation, power imbalance, and legal permissiveness created conditions where everyone made locally "rational" choices that collectively produced catastrophe.

More specifically, learners should recognize:
- How vulnerability (Leonora's immigration status, debt) functions as a structural constraint on behavior
- How identity and self-interest can make exploitation feel like opportunity (Max's worldview)
- Why reform is difficult even when danger is visible (Rosa's constraints)
- That "individual bad choices" explanations often obscure systemic dynamics

---

## 4. Key Interaction

**Setting:** The 9th floor cutting room, Triangle Shirtwaist Factory, Saturday afternoon, March 25, 1911. It is 4:30 pm — 30 minutes before the end of the shift. The factory is loud and crowded. Rosa has slipped in through the freight entrance during a delivery, disguised under a shawl. She finds Leonora at her table.

**The interaction:**
Rosa approaches Leonora quietly and begins talking about union membership — the union is fighting for a 52-hour work week, fire exits that open, and sprinkler systems. Leonora is visibly interested but afraid; she has a family to feed and a sister to bring over. As they talk, Max crosses the floor and notices Rosa — he knows her face from the 1909 strike. He confronts her.

**What the interaction makes visible:**
| Character | Revealed by the interaction |
|-----------|----------------------------|
| Leonora | She knows the conditions are dangerous (has noticed the locked doors). But she cannot afford to know — acting on that knowledge means losing the job that her family depends on |
| Max | He is not a simple villain. He frames everything through the lens of his own story: he made it through hard work; these workers can too; unions are outside agitators threatening what he built. He genuinely cannot see what Rosa sees |
| Rosa | She has a specific, concrete fear (the locked doors) and a specific, concrete ask (sign the card). Her urgency and the workers' hesitation make the structural barriers to change tangible |

The three-way conversation shows learners that the fire was preventable — Rosa is there, warning, with a solution — and that it happened anyway, not because no one cared but because the system of incentives and vulnerabilities made prevention nearly impossible.

---

## Technical Implementation

### Architecture

A Python simulation using the **Anthropic Claude API** (modern replacement for the deprecated OpenAI text-davinci models used in Stanford Town). Implements a simplified version of the Perceive → Plan → Converse cognitive loop.

Each agent is represented by:
- **Identity block** (name, age, background, traits, current goal)
- **Memory** (bootstrap memories as text — what they know, what they've seen, what they fear)
- **Constraints** (explicit list used to guide generation)

The simulation runs a multi-turn conversation among all three agents in the key interaction scene, with each agent's turn generated by a Claude API call with:
- System prompt: historical context + that agent's full identity/memory/constraints
- Human turn: the current state of the interaction (what was just said/done)

### Files to Create

```
Project/
├── simulation.py              # Main simulation script (CLI runnable)
├── simulation.ipynb           # Jupyter notebook version for submission
├── agents/
│   ├── leonora_russo.json     # Identity, memory, constraints
│   ├── max_bernstein.json     # Identity, memory, constraints
│   └── rosa_peretz.json       # Identity, memory, constraints
└── requirements.txt           # anthropic, jupyter
```

### Agent JSON format (mirrors Stanford Town's scratch.json pattern)
```json
{
  "name": "Leonora Russo",
  "age": 22,
  "innate": "hardworking, cautious, loyal, frightened",
  "learned": "...",
  "currently": "...",
  "bootstrap_memories": ["...", "..."],
  "constraints": ["...", "..."],
  "daily_plan_req": "..."
}
```

### Key functions in simulation.py

| Function | Purpose |
|----------|---------|
| `load_agent(path)` | Load agent JSON, return Agent object |
| `build_system_prompt(agent, scene_context)` | Construct the historical system prompt for a given agent |
| `agent_turn(agent, conversation_history, client)` | Call Claude API, return the agent's next action/speech |
| `run_interaction(agents, scene, turns)` | Orchestrate the multi-turn conversation, print transcript |
| `main()` | Load agents, set scene, run interaction |

### Critical files to reference during implementation
- Stanford Town scratch.json format: `/Stanford_Town/repo/environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/personas/Isabella Rodriguez/bootstrap_memory/scratch.json`
- Stanford Town persona.py for cognitive loop reference: `/Stanford_Town/repo/reverie/backend_server/persona/persona.py`
- Converse module for conversation structure: `/Stanford_Town/repo/reverie/backend_server/persona/cognitive_modules/converse.py`

---

## Verification

1. `pip install anthropic jupyter` (or check requirements.txt)
2. Set `ANTHROPIC_API_KEY` environment variable
3. Run `python simulation.py` — should produce a multi-turn transcript of the factory floor interaction
4. Check that each agent's dialogue reflects their historical constraints:
   - Leonora: expresses fear of losing job, hesitates before committing
   - Max: frames things through individual achievement, defends the factory
   - Rosa: references specific dangers (locked doors), makes the union ask
5. Run `jupyter notebook simulation.ipynb` for the notebook version
