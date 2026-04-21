---
title: "Reporting Agent Behaviors — Triangle Shirtwaist Factory, March 25, 1911"
author: "John Baker"
course: "EDUC_5919 — Deep Learning and Transformer Models"
date: "April 20, 2026"
---

# Reporting Agent Behaviors

### A three-agent generative simulation of the Triangle Shirtwaist Factory, 4:30 PM, March 25, 1911

**John Baker · EDUC_5919 · Spring 2026**

---

## 1. Historical World Overview

**Time and place.** Saturday afternoon, March 25, 1911 — the 9th-floor cutting room of the Triangle Shirtwaist Company, on the top three floors of the Asch Building at the corner of Washington Place and Greene Street in Greenwich Village, New York City. The simulation is pinned to the thirty-minute window beginning at 4:30 PM. At roughly 4:45 PM on this day, a scrap bin on the 8th floor ignited; by 5:00 PM the building was in full flame; 146 workers, mostly young Jewish and Italian immigrant women, were dead, either burned inside the building, trapped behind locked doors, or fallen from 9th-floor windows onto Greene Street.

**Social, political, and cultural context.**
- The shirtwaist trade had a workforce of roughly 40,000 in Manhattan in 1911, more than 80% of it female, more than 70% Jewish or Italian immigrant.
- Workers labored 52–72 hours a week for piece-rate wages. Factory owners routinely **locked stairwell doors during shifts** to prevent pilferage of cloth scraps and unauthorized breaks. No state law forbade the practice in 1911.
- The **1909 Uprising of the 20,000** — a 13-week general strike of shirtwaist workers led by Clara Lemlich and the ILGWU — had won contracts at many shops but failed at Triangle, whose owners (Isaac Harris and Max Blanck) paid for strikebreakers, hired police, and emerged un-unionized.
- Immigrant workers were economically and legally precarious: deportation for "public charge" was a live threat, and employers maintained an informal blacklist of organizers and agitators that followed a worker from shop to shop.
- The fire became, retrospectively, the event that broke the political logjam on industrial safety regulation in New York State (via the Factory Investigating Commission and the subsequent wave of labor legislation led by Frances Perkins and Al Smith). None of that exists yet on the afternoon of March 25, 1911.

**Important places in the world.** The simulation is geographically minimal by design: the scene is a single room (the 9th-floor cutting floor), with the Washington Place stairwell door (locked, every shift) and the Greene Street freight door (the only open exit, used by Rosa to enter) named as structural features. The 8th floor is named as the place where the fire will begin. This compression is deliberate — see §6.

**Why I designed the world this way.** The assignment brief asks for a world where understanding behavior *requires* attention to social, economic, and cultural context. The Triangle fire is the clearest case I know of a disaster that is routinely explained as a moral story — "owners were greedy, doors were locked, people died" — when the historical mechanism is structural: every party was making decisions that were locally rational within constraints (owners under competitive pressure and weak inspection, workers under debt and blacklist fear, organizers under legal and physical risk) and those locally rational decisions composed into catastrophe. Building this as a simulation lets students see the *local rationality* working in real time, which is invisible on the page when the story is told after the fact. The scene is compressed to the pre-fire half hour specifically so that the agents act with the knowledge they actually had that afternoon — not with hindsight.

## 2. Agent Setup

The simulation has three agents. Each is defined by a JSON file in `agents/` with the following fields, mirroring Stanford Town's `scratch.json` layering (`innate` / `learned` / `currently` — permanent traits → stable background → present goal) plus a flat `bootstrap_memories` list and an explicit `constraints` list. Identity is assembled into an **identity stable set (ISS)** string on each turn, a pattern ported directly from Stanford Town's `Scratch.get_str_iss()` in `memory_structures/scratch.py:382`.

### Leonora Russo — garment worker, 22

- **Background.** Sicilian immigrant, three years in New York. Shirtwaist cutter on the 9th floor. Speaks broken English; Sicilian at home. Main wage-earner for her family since her father's accident.
- **Goal.** Keep her job; save for her sister Concetta's steamship passage from Palermo.
- **Constraining forces.** Father's debt to a Mulberry Street loan shark; fear of the cross-shop blacklist; immigration precarity; distrust of outsiders.
- **Bootstrap memories** include: a neighbor fired last month for talking to a union woman; the Washington Place door being locked every shift; scrap bins that have smoked before; a cousin beaten during the 1909 strike; a photograph of Rosa passed around the union girls.

### Max Bernstein — floor supervisor and partial owner, 48

- **Background.** Jewish immigrant from Minsk. Worked his way up from cutter to foreman to owning a small stake in Triangle alongside Harris and Blanck.
- **Goal.** Meet the weekly output quota; prevent theft; keep organizers out.
- **Constraining forces.** Pressure from the majority owners; his own self-made-man identity, which makes collective action feel like a personal insult; fire codes he bends because enforcement is weak.
- **Bootstrap memories** include: his arrival at Castle Garden with $14; his rise to ownership; the 1909 strike from the owners' side; a scrap-bin fire the year before that was beaten out and not reported; being specifically warned by the office about Rosa Peretz.

### Rosa Peretz — ILGWU organizer, 24

- **Background.** From Białystok via pogrom. Former Triangle worker (8 months, 1908). On the picket lines for the full 13 weeks of the 1909 Uprising; arrested twice. Now paid by the ILGWU to organize Lower East Side shops.
- **Goal.** Get Leonora to sign a union card, because Antonia from Elizabeth Street named her as someone with sense.
- **Constraining forces.** Banned from the building by name. A third arrest would carry a heavier sentence. The ILGWU itself is male-dominated and skeptical of the women workers she is trying to sign.
- **Bootstrap memories** include: the locked Washington Place door (she tried the handle herself in 1908); the scrap bins; her friend Sarah's jaw broken by Policeman Brodsky on 14th Street; Max's face from 1909; the Elizabeth Street meeting where Leonora's name was passed.

### Prompt architecture

Each turn is a single OpenAI `chat.completions` call to `gpt-4o`. The system prompt carries:

1. A scene description (room, time, what's happening, what *none of them know*).
2. The agent's ISS string (`Name / Age / Innate / Learned / Currently / Daily plan / Current date`).
3. The **top-k memories retrieved** from the agent's `AssociativeMemory` stream for this turn — scored by `recency + relevance + importance`, a port of Stanford Town's `cognitive_modules/retrieve.py`. The stream is seeded at load with all bootstrap memories (each embedded with `text-embedding-3-small`); as the conversation runs, every utterance the other two agents make is written into the stream as an observation node with a model-rated importance score (1–10, via `gpt-4o-mini`), and reflections (below) are added as further nodes. Only the top-5 memories for this turn's retrieval query land in the prompt — not the full stream.
4. All of the agent's explicit behavioral constraints, one per line (including the hard rule that they *do not* know the fire is coming in 30 minutes).

The user message carries the running transcript of the conversation and asks "what does `<first_name>` say or do next? One to three sentences. In period voice. No speaker label." The turn order is scripted (Rosa → Leonora → Rosa → Leonora → Max → Rosa → Max → Leonora → Rosa → Max), which is a deviation from Stanford Town's dynamic converse module but appropriate for a single-scene historical set-piece.

**Reflection.** When an agent's accumulated observation-importance crosses a threshold (18, tuned to fire once or twice per agent in a ten-turn scene), reflection triggers at the top of that agent's next turn — *before* retrieval runs, so fresh insights are eligible for the top-5. The model is asked for two first-person insights ("I see that...", "I am beginning to realize...") drawn from the agent's most recent memories; each insight is embedded and appended to the stream as a new node with `source="reflection"` and elevated importance, then the trigger resets. This is a port of Stanford Town's `cognitive_modules/reflect.py`, simplified: most-recent-N pool instead of focal-question generation, no reflection-of-reflections. See `memory.py` and §3.6 below for an empirical example.

Three runs were executed end-to-end after the retrieval + reflection retrofit (`logs/run_20260420_212326.json`, `_212427.json`, `_212527.json`). Behavior observations §3.1–§3.5 below were drawn from an earlier round of runs using a simpler architecture (all bootstrap memories passed directly into every prompt); those observations carry forward because the specific utterances they cite are unchanged. §3.6 documents a behavior that only becomes visible once reflection is running and cites specifically to the post-retrofit logs.

---

## 3. Behavior Documentation

The simulation produced a first round of runs with the simplified architecture (all bootstrap memories passed into every prompt) and a second round of three runs after the retrieval + reflection retrofit described in §2. Below are six specific behaviors worth documenting, each cited to a specific line and run. The full JSON logs are attached to the submission; §3.6 is drawn from the post-retrofit logs.

### Observation 1 — Code-switching under stress (Leonora)

On every run, Leonora's English becomes less fluent exactly when she is most frightened. She drops Italian and Sicilian phrases at moments of social danger and almost never at neutral moments.

> **Run 2 / turn 2 (Leonora):** "Signorina, per favore, I work here. I cannot be seen with... with you. They fire Maria last month." *she lowers her voice, eyes darting to the foreman*
>
> **Run 3 / turn 4 (Leonora):** "If I sign, they know. Maria, she no work after... **Io ho paura, capisci?**" [trans.: *I am afraid, do you understand?*]
>
> **Run 2 / turn 4 (Leonora):** "Signora Rosa, **io**... I want to help, but I need this job. For my **famiglia**."

This is not a scripted behavior — the constraint says only "your English is broken; you use simple words, sometimes Italian or Sicilian phrases." The model is using language-switching *as a stress signal*, which is documented in the sociolinguistic literature on immigrant speakers (Zentella 1997 on Spanish-English code-switching; for Italian-American speech specifically, Haller 1993). The simulation is doing something linguistically plausible that I did not specifically program.

### Observation 2 — Rosa's physical concealment of the union card

In two of three runs, Rosa physically **hides** the card rather than openly handing it over — once by slipping it onto the table mid-cuttings, once by tucking it under a fabric scrap. Neither maneuver is in her constraints; what the constraints specify is "you are trespassing and in real danger of being caught" and "you are not here for a speech — you are here to get one card signed."

> **Initial run / turn 9 (Rosa):** "...she nods to Leonora, quietly slipping a union card onto the table amid the cuttings. For later, when you have time to think."
>
> **Run 3 / turn 9 (Rosa):** "...Rosa slips the card and pencil under a scrap of fabric on the table before moving away, muttering softly. 'Think of your future, Leonora. We will talk again soon.'"

This is **emergent instrumentality** — the model has inferred that her goal (card signed) and her constraint (must not be caught) imply a plausible physical workaround (drop the card, leave). It is also a tactic that ILGWU organizers in period accounts actually used. It was not in the prompt.

### Observation 3 — Max's ideological framing of the locked doors

In every run, when doors come up, Max frames the locking as *productive*, not as a safety issue. This is strikingly consistent:

> **Initial run / turn 10 (Max):** "Safety comes from doing your job right, not from listening to agitators."
>
> **Run 2 / turn 10 (Max):** "Locked doors protect what we build here, Rosa."
>
> **Run 3 / turn 7 (Max):** "Out! And take that business to the street! This is not the place for your speeches."

The constraints describe Max as "not a cartoon villain. When you defend the factory, you mean it." The output shows him doing exactly that — answering a safety critique with a productivity frame, not hearing the critique on its own terms. This is the **worldview-as-constraint** pattern the simulation was designed to expose: self-interest is not experienced as self-interest by the person holding it.

### Observation 4 — The blacklist invoked by name

In all four runs, Leonora names the specific consequence of unionizing — being fired, being on "a list" — by reference to a named past coworker ("Maria") that the bootstrap memory attributes to her.

> **Run 1 / turn 4 (Leonora):** "The foreman, he watch us."
>
> **Run 2 / turn 2 (Leonora):** "They fire Maria last month."
>
> **Run 3 / turn 4 (Leonora):** "If I sign, they know. Maria, she no work after..."

The bootstrap memory reads: "Last month, Maria from Bleecker Street was fired for talking to a union woman in the cloakroom. The next week, she could not get work at any of the shops on Broadway. Her name was on a list." The agent is not paraphrasing or summarizing this memory — it is **invoking it as evidence** in argument with Rosa, which is exactly how memory-in-conversation works in real human dialogue. This is the pattern Park et al. (2023) called "memory retrieval in service of current action." In the pre-retrofit runs quoted above, every bootstrap memory was passed into every prompt, so the model could reach for this one freely; in the post-retrofit architecture (§2) the retrieval module scores and pulls the top-5, and the audit trail in `logs/run_2026*.json` (`retrieved_memories` field per turn) confirms the Maria memory consistently ranking into Leonora's top-5 on the turns where she invokes it.

### Observation 5 — Rosa's last line

In every run, Rosa uses her final turn not to escalate against Max but to send one more message to Leonora. This is consistent with her constraints ("you are not here for a speech — you are here to get one card signed") but the *choice of what to say* is emergent.

> **Initial run:** (no final Rosa line — she was cut off by Max at turn 10)
>
> **Run 1 / turn 9 (Rosa):** "We'll find you again, cara. Be strong and remember, there's a way to change this."
>
> **Run 2 / turn 9 (Rosa):** "I understand, Max. I'll not cause trouble today. But remember, locked doors don't sew the line either." — *a deflection at Max that is actually a message to Leonora*
>
> **Run 3 / turn 9 (Rosa):** "Think of your future, Leonora. We will talk again soon."

### Observation 6 — Max privately concedes what he publicly denies (post-retrofit)

With reflection running, the simulation produced a behavior that was not visible in the earlier runs: Max's *internal* position on locked doors diverges from his *spoken* position. The reflection module, which is asked only for first-person insights drawn from the agent's recent observations, twice generated the same structural concession for Max at his fourth turn:

> **Run 20260420_212326 / turn 4, Max's reflection (not spoken):** "It is becoming clear to me that strict rules like locking doors are enforced to prevent theft but may unintentionally create unsafe working conditions."
>
> **Run 20260420_212527 / turn 4, Max's reflection (not spoken):** "It is becoming clear to me that the locked door policy, intended to prevent theft, also creates an unsafe environment that could have dire consequences."

Compare to what Max *says* on the same turns: "What is this? You again, Miss Peretz? I told you, these young ladies have work to do, not time to listen to your shtik" (212326 / turn 4) and "Concerns, eh? We give fair wages for fair work. You bring your trouble elsewhere" (212527 / turn 6). Then, on turns after the reflection has been written into his memory stream, the retrieval audit shows the insight being pulled into subsequent prompts (see `retrieved_memories` at turns 6 and 7 in `logs/run_20260420_212527.json`) — available to the model as context, but not spoken aloud.

This is a **second-order** behavior. The simplified architecture could not have produced it: with all bootstrap memories flat in the prompt, there was no distinction between what an agent had "noticed" and what they were willing to say. Adding reflection (a port of Stanford Town's `reflect.py`) created the separation — an internal position that influences the next reasoning step without surfacing as speech. It is also historically resonant: Harris and Blanck's trial testimony in December 1911 showed that owners understood the door policy created risk and chose to continue it for commercial reasons. Max's unspoken reflection follows exactly the same logic. I did not prompt for this; the reflection module generated it from his recent observations of the Rosa/Leonora exchange about locked doors.

---

## 4. Behavior Analysis

### Patterns

Three patterns held across all runs:

1. **Power reveals through action, not speech.** Max does not make arguments for why he has the right to eject Rosa; he just ejects her. "Out! Out now!" "Move!" Rosa, by contrast, *argues* — she names specifics, she bargains, she appeals. This asymmetry of *who has to justify themselves* is more or less exactly the asymmetry the historical record describes. It was not specified in the prompt.

2. **Leonora does not refuse. She defers.** In none of the four runs does Leonora say "no" to signing. She says "maybe later," "not now," "for the famiglia," "I want to help, but..." The economic-vulnerability constraint is producing something subtler than refusal — it is producing the specific conversational move that says *I cannot afford to decide this*. This is what the simulation was designed to surface and is, in my reading, the single most important thing it produces.

3. **Rosa's specificity holds up under interruption.** The constraints insist Rosa make concrete asks (sign the card; ten cents a month; unlocked doors; sprinklers), and the output respects this — she does not drift into rhetoric even when Max is shouting at her. Compare her final lines across runs: each is a small, tactical message, not a speech.

### How environment, prompts, and roles shaped outcomes

The scene context does a large amount of work. The prompt names the locked Washington Place door and the open Greene Street freight door in the first paragraph, and in every run those two doors reappear in the dialogue — Rosa uses the locked-door argument, Max defends it, Leonora glances at it. Features of the *environment* that the agents were told about showed up in their talk. Features I did *not* specify (e.g. "what is the floor lit by" — gas? electric? late-afternoon sun through the 9th-floor windows?) never came up. The simulation is reflective, not generative, of its prompt environment.

The `constraints` list did much of the heavy lifting for period voice. When Rosa's constraint included "you may use a Yiddish word occasionally," `nu` appeared in her speech (Run 3 / turn 5, Max: "*Nu*, what's this?" — actually Max's, showing the constraint applied consistently to the Jewish-immigrant agent). When Leonora's constraint named specific Italian phrases, those phrases showed up. Removing the constraint would almost certainly cause the agents to drift into modern American English, because that is the model's default.

### Realistic vs. artificial

Realistic:

- The rhythm of confrontation (Rosa whispers, Leonora hedges, Max enters and escalates the register, Rosa de-escalates and withdraws).
- The asymmetry of vulnerability (Rosa risks arrest; Leonora risks her job and her sister's passage; Max risks nothing).
- The refusal to name the future (nobody predicts the fire, as the constraint specifies; but also, nobody says "something terrible is coming" — they are in the moment, as the constraint specifies).

Artificial:

- Every exchange is legible, coherent, and moves forward. Real factory-floor exchanges in 1911 would have been frequently cut off by the noise of the machines, by other workers needing to pass, by Leonora being interrupted mid-sentence by a foreman asking about a piece count. The simulation gives each agent clean turns with no interference.
- Nobody swears. Nobody is unclear. Nobody loses the thread.
- The agents' language is *too* grammatical, even when it is supposed to be broken. Leonora's "My English is not good" reads as a fluent speaker *performing* non-fluency. This is a well-known failure mode of LLM dialect rendering.

### What surprised me

The clearest surprise was Observation 2 — Rosa's physical concealment of the card. Nothing in the prompt tells her *how* to give Leonora the card. The model inferred a tactic (leave it, don't hand it over) that is consistent with both her goal and her constraints and that matches period accounts of covert organizing. This is the kind of small, context-appropriate inference the Park et al. (2023) paper argues generative agents are capable of, and I had not expected to see it so cleanly in a three-agent scene without the full cognitive loop.

The other surprise, in the opposite direction: how thoroughly the agents *avoided* the moment the simulation is about. None of them mentioned the fire. None of them lingered on the locked door beyond the argument. This is the correct output (they were told they do not know the fire is coming), but it made the scene feel *ordinary* — which is, perhaps, the most historically important thing the simulation produces. Nobody on the 9th floor of the Asch Building at 4:30 PM on March 25, 1911, thought they were living in a catastrophe. They thought it was Saturday.

---

## 5. Historical Plausibility Evaluation

### What fits

- **Locked stairwell doors.** Accurate. The locking of exit doors to prevent theft was a documented and widespread practice; survivor testimony at the trial of Harris and Blanck in December 1911 confirmed the 9th-floor Washington Place door was locked at the time of the fire.
- **Immigrant demographics.** Accurate. The Triangle workforce was approximately 75% Jewish and 20% Italian immigrant, overwhelmingly female, mostly aged 16–24 — Leonora's age is typical; Rosa's age and trajectory (Uprising of the 20,000 → ILGWU → covert organizing) match organizer profiles of the period.
- **The blacklist.** Accurate. Inter-shop sharing of organizer names is documented both by organizer testimony (Pauline Newman, Clara Lemlich) and by internal employer correspondence.
- **The 1909 strike.** Accurately placed as a reference point for both Max and Rosa. Their different relationships to it (Max: "the strike was broken and the shop stayed open"; Rosa: "At Triangle, we did not win. The girls went back. The doors stayed locked.") reflect the actual asymmetry of how the strike was remembered on each side.
- **Max's trajectory.** Plausible but not historically grounded on a real person. The composite — Jewish immigrant, worked up from cutter to partial owner — matches what is known about Triangle's ownership (Isaac Harris was such a figure, though with a much larger stake; Max Blanck too). Max Bernstein himself is a composite character.

### What's elided or anachronistic

- **Language.** The agents' English is too fluent, including when it is supposed to be broken. A real Leonora in 1911 likely could not have held this conversation in English at all. The prompt acknowledges this but the model cannot fully enact it.
- **Awareness of the ILGWU by immigrant Italian women.** Leonora's recognition of "a woman named Rosa" is plausible but probably overestimates the penetration of union consciousness into the Italian workforce in 1911. The Italian garment workers were notoriously *under*-organized relative to the Jewish workforce, partly for language reasons, partly for religious reasons (hostility from Catholic clergy toward what was seen as Jewish radicalism), and partly because many Italian women planned to return to Italy and did not invest in American civic institutions.
- **Nobody is hostile to Rosa besides Max.** In a real Triangle floor, other workers — some of them committed to keeping their jobs — might have reported her to the foreman before Max saw her. The simulation's two-and-a-half party structure (Rosa, Leonora, then Max) is cleaner than the historical reality of a crowded factory floor.
- **The compressed scene.** The real pre-fire interval on the 9th floor was filled with the last rush of Saturday work — piece counts, supply runs, the sounds of the 8th floor below, mundane conversation in three languages. The simulation stages a single focused encounter. This is a dramatic compression, not a historical description.
- **Gender and authority.** Max speaks to Rosa and Leonora the same way in the transcripts, but historically a 48-year-old male supervisor would have addressed a 24-year-old organizer and a 22-year-old worker in subtly different registers. The model does not pick up this differentiation.

### What feels missing

- **The other 397 workers.** The 9th floor had roughly 400 people on it at 4:30 PM. The simulation has three. The scene reads as if the room were empty except for the speakers.
- **Religious and familial obligation.** Leonora's memories name her priest ("Padre Alessandro at Old St. Patrick's says to work hard and trust God") but this never enters her dialogue. In period, religious counsel was a significant brake on radicalism in Italian immigrant communities and would almost certainly have been present in Leonora's deliberation.
- **Material sensation.** No dust, no thread, no sewing-machine hum from below, no sweat. The simulation is a conversation, not a place.

### Anachronisms

No direct anachronisms (OSHA, 40-hour week, modern labor vocabulary) appeared in any of the four runs. This is a genuine win for the explicit constraint: "Stay in period voice. No references to OSHA, the 40-hour week, modern labor law, the later history of the Triangle fire, or anything after March 1911." Without that constraint in the prompt I expect at least one run would have produced an anachronistic phrase.

---

## 6. Simulation, Context, and Critique

**The simulation is not a history lesson. It is one component of a learning process whose core is elsewhere.**

### Prior knowledge students would need before engaging

A student who arrived at this simulation cold would derive almost nothing useful from it. Before running it, they should have engaged with:

- **The factual record of the fire itself** — number of deaths, known causes, the locked-door finding, the trial and acquittal of Harris and Blanck. Leon Stein's *The Triangle Fire* (1962) remains the standard account; David Von Drehle's *Triangle: The Fire That Changed America* (2003) is a good alternative.
- **The 1909 Uprising of the 20,000** — without this, Rosa is illegible. Annelise Orleck's *Common Sense and a Little Fire* (1995) covers it.
- **The structure of the shirtwaist industry** — piece-rate wages, subcontracting, the position of immigrant women in the early-20th-century Manhattan garment trades.
- **The concept of structural explanation in history** — that causation in historical events is often a composition of many individual rational choices under constraint, not the action of a villain. Without this framing, students will read Max as a simple antagonist and miss the point.
- **The history of American immigration, specifically Italian and Eastern European Jewish immigration c. 1880–1914** — Leonora and Rosa both make assumptions (about family, about debt, about return migration, about pogroms) that require context to land.

### Why the simulation is not standalone

The simulation cannot teach history because:

1. **It generates possible behavior, not evidence.** Nothing Leonora, Max, or Rosa says in any run is attested. They are composite plausible characters. Students who do not know the difference between *this is what someone in this position might have said* and *this is what someone in this position said* will confuse the two.
2. **It flattens what was messy.** A four-agent (and historically, 400-agent) floor with language barriers, noise, physical exhaustion, and unclean turn-taking is rendered as a three-way conversation with clean turns. The clarity is part of the pedagogical use — you can see the dynamics — but it is also part of the distortion: the dynamics feel more legible than they were.
3. **It enforces a specific reading.** By naming "the system of incentives and vulnerabilities" in the constraints, and by writing memory fields that connect Max to 1909 and Leonora to the blacklist, I have committed the simulation to a structural interpretation of the fire. This is the interpretation I endorse and that most working historians of the event endorse, but it is still an interpretation. The simulation performs this interpretation; it does not discover it.
4. **It is epistemically opaque.** A student cannot tell which of Max's lines are grounded in the historical record, which are generated by the model's prior, and which are artifacts of the prompt. Without external scaffolding, the student has no way to calibrate trust.

### Assumptions the simulation makes

- That structural forces (economic desperation, worldview, legal constraint) are the primary explanatory frame for the fire. This is a choice, not a neutral observation.
- That the three selected agents (worker, supervisor, organizer) are the right spine for the story. A fourth agent — a fire inspector, a factory owner (Harris or Blanck), a journalist, a policeman, a mother at home — would foreground different forces.
- That the critical scene is an interpersonal confrontation at 4:30 PM. An alternative simulation might run for three weeks and show Rosa's recruiting campaign, or for six hours and show the fire itself. The choice of scene structures what is thinkable.

### What the simulation gets wrong, flattens, or fails to represent

- **Scale.** Three agents stand in for 400 people on the floor (and for the tens of thousands in the trade).
- **Embodied reality.** No physical sensation, no exhaustion, no hunger, no pain from a 13-hour shift.
- **Language texture.** Over-fluent broken English; no untranslated Italian/Yiddish passages; no noise.
- **Time.** Agents can speak at conversational length in a scene that, realistically, would last under 90 seconds before Max arrived.
- **The aftermath.** Nothing in the simulation marks that everyone in this room will be dead, or have watched their friends die, in 30 minutes. The simulation is structurally barred from referencing what gives the scene its meaning. This is correct (the agents cannot know) but it is also limiting (the student watching cannot *not* know, and the simulation does not help them hold both).

### How I would design learning activities around this

- **Before:** assign Stein or Von Drehle's account of the fire; a selection of primary sources (Clara Lemlich's Cooper Union speech, Pauline Newman's oral history, survivor testimony); and a short piece on the difference between structural and proximate causation in historical explanation.
- **During the simulation:** have students run it two or three times. Have them annotate each transcript line with "which part of this agent's background does this line draw on?" — a forced reading against the agent JSON. Have them mark any line that strikes them as anachronistic or implausible.
- **After:** a close-reading exercise in which students compare a specific line in the simulation to a specific attested document (e.g. Rosa's line about locked doors vs. Frances Perkins's famous remark after watching the fire from Washington Square; Max's self-made-man framing vs. Isaac Harris's trial testimony). The question is: *where does the simulation echo the record, and where does it invent?* This is where the critical interpretive work happens.
- **Assessment prompt:** not "what did you see in the simulation?" but "what did the simulation help you *see* in the historical record — and what did it mislead you about?"

The simulation is a tool for *activating* historical imagination, not for *informing* it. It must come after context and before critique. Used alone, it produces confident, plausible, invented dialogue that students will over-trust.

---

## 7. Strengths and Limitations

### What the simulation helps reveal

- **Local rationality.** It is genuinely difficult to read the transcripts and come away thinking Max is a villain or Leonora is weak. Each of them is responding sensibly to the forces they are embedded in. This is the single most pedagogically valuable outcome.
- **The asymmetry of risk.** Who has to argue, who has to leave, who has to keep their head down — this is visible in the *structure* of the dialogue, not just its content. Students learn to read power relations off a transcript.
- **Memory-in-argument.** Leonora's invocation of Maria, Rosa's invocation of Sarah and Brodsky, Max's invocation of the 1909 strike and his own arrival — the agents deploy personal history as rhetoric. This is how real people argue, and it is a useful thing to see modeled.

### What the simulation cannot capture

- **Embodiment.** As above.
- **Silence.** The simulation has to produce an utterance on every turn. Real conversation in a factory would have long silences, half-sentences, nods, gestures. These are invisible.
- **The full cast.** Three people is a narrow window into a 400-person room in a 40,000-worker industry in a city of 5 million.
- **Outcome.** The simulation ends with Max telling workers to get back to work. In real life, the fire begins 15 minutes after this scene closes. The simulation is structurally unable to surface that this is an ordinary moment *because* disaster is coming — the hindsight weight that gives the scene its meaning is outside the scope of what the agents can access.

### Engagement, exploration, or critique?

All three, unequally:

- **Engagement: high.** The transcripts are dramatic and emotionally legible. Students will care about Leonora within five lines.
- **Exploration: moderate.** By re-running and varying prompts, a student can ask "what if Rosa were more aggressive? What if Leonora had no sister to save for? What if Max had lost his daughter to a factory accident?" — and see the system respond. This is genuinely useful for counterfactual reasoning.
- **Critique: the highest value, but only if framed.** The simulation is useful as an *object* of critique more than as a *source* of knowledge: what does this AI produce when asked to play a historical subject, and what does its production reveal about the limits of generative modeling? That is the deepest lesson here, and it is also the lesson the student will not reach without a teacher.

---

## 8. Final Reflection

What did analyzing agent behavior in a historical world help me understand about both history and AI systems?

**About history.** The value of the simulation was not in any new fact about the Triangle fire. I knew the facts going in; the JSON files encode them. The value was in watching the facts *interact* — in seeing Leonora's father's debt appear in her mouth as a reason not to sign, in seeing Rosa's 1909 memory come out as a concrete argument about locked doors, in seeing Max's immigrant-success story deployed against a labor critique. Historical explanation as I learned it was static: *these forces produced this outcome*. Watching agents argue inside those forces made the forces feel *active* — not as a diagram but as pressure on a choice. That is pedagogically different, and I think valuable.

I also learned something specific to this event. I had filed the Triangle fire mentally as a safety-regulation story. The simulation reoriented me: it is also, and maybe primarily, a labor-organization story. Rosa cannot win in the scene not because sprinklers are not invented (they are, in 1911) and not because the exit doors cannot be unlocked (they can, trivially), but because Leonora cannot afford to sign. The structural constraint operates on the worker more than on the owner. That asymmetry — that the only way to change the doors is for the workers to refuse to come in, and that the workers cannot refuse to come in — is the actual mechanism of the disaster. The simulation made this visible in a way reading about it did not.

**About AI systems.** Three things stood out:

First, the degree to which the model *performs* the prompt. Everything the agents do is downstream of what I put in the JSON files and the system prompt. When I specified Italian phrases, Italian phrases appeared. When I specified no anachronisms, no anachronisms appeared. When I did *not* specify physical environment (dust, noise, light), the agents acted as if in a blank room. This is the inverse of how human historical imagination works: a human historian fills in what is not specified; a language model produces approximately what it is prompted to produce. The skill of building a good simulation is almost entirely in the prompt architecture, not in the run.

Second, the model's real strength is **composition**. Each agent's bootstrap memories are a set of unrelated facts. The model combines them into arguments. Rosa's memory of Sarah's broken jaw combines with her memory of Brodsky combines with her goal of getting a signature to produce: "*I know it's dangerous, Leonora. But our danger now is greater.*" None of that sentence is in the memories verbatim. The model is building arguments out of parts. That is a genuine capability, and it is the thing that makes these simulations feel alive.

Third, the model's characteristic weakness is **over-coherence**. Real historical subjects contradicted themselves, forgot things, changed the subject, misunderstood each other, and failed to argue cleanly. The model's agents do none of these. They are always articulate. They are always on topic. They are always saying something that moves the scene forward. Real people often are not. The *limits* of what a generative agent simulation can teach about history are set, I think, here: it cannot teach what incoherence feels like, and a lot of lived history is incoherent.

Fourth — and this is what the reflection retrofit (§3.6) made visible — generative agents can be built with a **second-order layer** that is qualitatively different from their first-order speech. When Max's reflection module generated "the locked door policy, intended to prevent theft, also creates an unsafe environment," and he never said so aloud, I was watching the simulation do something I had not seen it do in the pre-retrofit runs: hold an interior position that diverged from its public position. Park et al. (2023) frame reflection as a mechanism for long-horizon memory compression. What I saw instead was reflection functioning as a mechanism for modeling *dissimulation* — the gap between what a historical actor understood and what they said. That gap is structurally important to the Triangle story (Harris and Blanck's trial testimony shows it in the record) and is exactly what a first-order dialogue simulation cannot produce. The generalization: the cognitive modules Park et al. built for game-like open-world agents happen to also be useful for modeling the interiority of historical subjects, which is a different use case than the one they were designed for but maps onto it cleanly.

Taken together: generative agent simulations of historical worlds are useful as a **rehearsal of structural reasoning** — a way to see forces act on choices — and they are misleading as a **source of historical content**. The pedagogical challenge is building a learning environment that holds both of those at once. That is harder than building the simulation itself, and it is where the serious instructional design work lives.

---

## Appendices

- **A. Code.** See `simulation.py` (CLI), `simulation.ipynb` (notebook), `memory.py` (associative memory + retrieval + reflection), `agents/*.json` (agent identities).
- **B. Simulation logs.** Pre-retrofit runs (simplified architecture — all bootstrap memories flat in each prompt) cited in §3.1–§3.5 carry the earlier timestamps. Post-retrofit runs (retrieval + reflection running) cited in §3.6: `logs/run_20260420_212326.json`, `logs/run_20260420_212427.json`, `logs/run_20260420_212527.json`. Each post-retrofit log contains the timestamp, models, full system prompts, turn order, per-turn `retrieval_query` / `retrieved_memories` / `reflection` audit trails, the transcript, and the final memory streams for all three agents.
- **C. References to Stanford Town.** Identity-stable-set string: `Stanford_Town/repo/reverie/backend_server/persona/memory_structures/scratch.py:382-414`. Iterative conversation template: `Stanford_Town/repo/reverie/backend_server/persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt`. Agent JSON reference: `base_the_ville_isabella_maria_klaus/personas/Isabella Rodriguez/bootstrap_memory/scratch.json`. Associative memory: `reverie/backend_server/persona/memory_structures/associative_memory.py`. Retrieval scoring: `reverie/backend_server/persona/cognitive_modules/retrieve.py`. Reflection: `reverie/backend_server/persona/cognitive_modules/reflect.py`. Upstream repository: <https://github.com/joonspk-research/generative_agents>.
- **D. Historiographical references.** Stein, Leon. *The Triangle Fire* (1962). Von Drehle, David. *Triangle: The Fire That Changed America* (2003). Orleck, Annelise. *Common Sense and a Little Fire: Women and Working-Class Politics in the United States, 1900–1965* (1995).
