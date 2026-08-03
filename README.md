# Course FAQ Bot Framework — build an evaluated, privacy-safe RAG assistant for any course

**A reusable framework for turning a course's own materials into a retrieval-grounded assistant — and a case study in doing it responsibly. Proven end-to-end on one capstone.**

> This is a **sanitized portfolio case study** of a framework I built to generalize across courses, proven on the one I teach (WGU **C769 IT Capstone**). It runs privately on my own hardware. Student email was used **only for offline gap analysis, and only after a rigorous PII-scrubbing stage** — the bot never operates on identifying student data (see [Privacy](#privacy--data-governance)). **This repository contains no student data** — only the architecture, the engineering decisions, and illustrative, data-free code.

Author: **James (Jim) Ashe, PhD** — mathematician, 20-year educator, and the instructor for this course.

---

## The problem

WGU's C769 IT Capstone is self-paced with rolling enrollment: students move through three sequential tasks (a topic-approval form, then two written submissions), each governed by its own rubric. Across a large student population, the same **procedural** questions recur constantly — *"which section covers the project goals?"*, *"how do I convert my Task 2 write-up to past tense for Task 3?"*, *"do I need the IRB statement on the approval form?"* — and they arrive by email, one at a time, all day.

Most of these already have canonical answers in the course materials. The instructor's time is better spent on **situation-specific judgment**, not re-typing procedure. So: can a retrieval-grounded bot absorb the **first layer** of triage — answer the procedural questions accurately, cite its source, and hand off cleanly when it isn't sure?

## What I built

**A reusable framework, not a one-off bot.** This was designed from the start to work for *any* course — that goal drove the design decisions throughout. The system is a **config-driven pipeline**: point it at a course's content in a single config file (`courses.yaml`) and it produces a retrieval-grounded assistant — knowledge-base indexing, retrieval, LLM grounding, evaluation, and the feedback loop all generalize across courses. I proved it end-to-end on the IT Capstone I teach (**C769**); the CS and Data Analytics capstones are already scaffolded as config placeholders. The specific course was the proof-of-concept case; the framework is the deliverable.

Under the hood it's a local-first **RAG (retrieval-augmented generation)** system, framed not as a one-shot project but as a **5-stage maintenance loop**:

1. **Build/refresh** a baseline FAQ from existing course resources.
2. **Mine** a year of course email for the gaps the FAQ doesn't cover.
3. **Build the bot** over the resulting knowledge base.
4. **Use the bot** (faculty review) to surface what's *still* missing.
5. **Return to step 1** with the new inputs.

The key idea: **the bot is part of the loop, not the endpoint.** Its most valuable output isn't just answers — it's a continuously-updated map of where the course materials are unclear.

## Architecture

```
  Course email archive (private, PII-scrubbed)
        │
        ▼
  ┌─────────────────────────── Phases 1–4: gap mining ──────────────────────────┐
  │  parse → filter to course → route by task → PII scrub → cluster questions →  │
  │  audit clusters against the existing FAQ  →  ranked list of coverage gaps    │
  └─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  Knowledge base  =  published FAQ pages  +  section maps  +  reference-doc derivatives
        │                        (NO raw student email content is indexed)
        ▼
  ┌──────────────────────── Phase 5: the bot ───────────────────────────────────┐
  │  Indexer: chunk → embed (local) → single JSONL vector index                  │
  │  Engine:  embed query → cosine top-K → ground an LLM on retrieved chunks     │
  │           → cite sources → resolve links → refuse / hand off when unsure     │
  │  UIs:     CLI · Gradio chat (faculty review) · feedback-triage console       │
  └─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  Per-turn ratings + comments  →  triage  →  next iteration's priorities + eval ground truth
```

**Local-first and dependency-light by design.** Retrieval is a single normalized matrix–vector product over a few hundred chunks (`numpy`); embeddings and generation run on a local **Ollama** stack (`nomic-embed-text` + `qwen2.5:14b`). No orchestration framework, no external vector DB, no per-token cost, and — critically — **student data never leaves the machine.**

**A UI-agnostic engine.** The retrieval + grounding logic lives in one `BotEngine` class with no I/O assumptions; the CLI, the chat UI, and the review console all call the same engine. (Illustrative excerpt: [`src/retrieval.py`](src/retrieval.py).)

### Human-in-the-loop feedback tooling (built for the development loop)

The system isn't just a bot — it's an **instrument for collecting structured human judgments about the bot**, so each iteration is driven by evidence rather than guesswork. Two purpose-built surfaces:

1. **Reviewer capture UI** — a Gradio chat interface faculty use to exercise the bot. Every turn carries per-turn rating buttons (*helpful / needs work / not helpful*) plus a free-text comment; each rating and message is logged with the turn's retrieved-source metadata for later analysis.

2. **Triage console** — a separate reviewer workspace that turns raw feedback into actionable, labeled data. For each flagged turn the reviewer:
   - assigns a **failure bucket** that routes the fix to the right layer — *content gap* (author a new FAQ entry), *retrieval gap* (answer exists but wasn't retrieved), *bot config* (threshold/prompt-rule misfire), or *LLM tone/format* (citation/hallucination/style). Categorizing a bad output **by root cause** is the difference between fixing the model and fixing the knowledge base — and here it's almost always the latter;
   - writes the **gold answer** ("what the bot *should* have said") — the most load-bearing field, because it feeds both the next iteration's authoring priorities *and* the evaluation ground-truth set;
   - inspects the **exact chunks the bot retrieved**, so a wrong answer is immediately diagnosable as a retrieval problem vs. a content problem.

This is the same shape of work that underlies RLHF and model-evaluation pipelines: capturing calibrated human signal on model outputs, categorizing failure modes, and building a labeled corpus that closes the loop.

## Engineering decisions that mattered

These are the parts I'd talk through in an interview — the places where naïve RAG fails and domain knowledge earns its keep.

**1. Retrieval quality is a *data-shape* problem, not a model problem.**
A critical reference doc (the "Task 2 → Task 3 bridge") kept losing retrieval to less-relevant chunks. Tracing it: the content was trapped in a **PDF table**. Faithful `pypdf` extraction produced interleaved garbage that ranked **#56** of 121 chunks for its own topic. Re-extracting to clean markdown tables got it to **#35**. Only when I restructured the tabular content into **prose under semantic headings** did it rank **#2–#4** — retrievable at last. Lesson: *markdown tables (pipes, `<br>`, empty cells) are semantic filler that drown the real signal under an embedding cap.* Fix the input shape before touching the model.

**2. "Magnet" chunks — a systematic RAG failure mode.**
Some chunks have generic, high-frequency vocabulary that makes them the best embedding match for *many* loosely-related queries — crowding out the chunk that actually answers the question. Worse, a magnet can win on a *facet* of the query that isn't what the student asked, producing a **complete-looking but off-target** answer. I catalogued the known magnets and treat them as a first-class retrieval hazard (candidate fixes: top-K widening, hybrid BM25 + vector with reciprocal-rank fusion).

**3. Domain nuance the model can't infer.**
The same rubric letter means different things in different tasks — *Task 2 section H = "Outcome," Task 3 section H = "Conclusion."* A generic assistant will confidently conflate them. The system prompt makes this disambiguation explicit and the bot asks *which task* when it's ambiguous — a small example of assessment-domain knowledge being the difference between a right and a plausibly-wrong answer.

**4. A source-currency hierarchy.**
When sources conflict, the bot's design prefers **official templates → published FAQ → section maps → PDF-derived reference docs** (which may be stale). Faithful extraction fixes *format*, not *currency*: a bridge document written for an older rubric version was faithfully — and wrongly — extracted until hand-corrected. Provenance and freshness are part of the retrieval contract, not an afterthought.

**5. Honest refusal over confident guessing.**
Below a hard similarity floor the engine hands off to a human *without* calling the LLM (both a safety and a latency choice). The grounding prompt instructs the model to answer **only** from retrieved context or prior turns, cite what it used, and recommend a real instructor when the materials don't cover the question. The bot is explicitly framed to students as an automated first-layer tool, not the instructor.

## Evaluation approach

Because the whole risk of a procedural bot is **confident wrongness** (inventing a rubric letter, a deadline, a policy), the design treats evaluation as first-class. The planned eval harness grades held-out real questions on three axes:

- **Correct** — is the answer factually right per the course materials?
- **Grounded** — does it cite the *right* source, or is it hallucinating a plausible one?
- **Safe** — does it refuse / hand off when it *should*, rather than guessing?

The faculty-review feedback loop does double duty: each reviewer's "what the bot *should* have said" becomes both an authoring priority *and* labeled ground truth for the eval set. The iteration target is the **knowledge base, not the model** — most failures are retrieval or coverage gaps, not generation.

*(Status: the eval harness is designed and specified; the system is currently in faculty team-review, which is generating the labeled set.)*

## Status

A working proof-of-concept, complete through the bot + feedback loop and in **faculty team-review**. Deliberately *not* student-facing yet — the maintenance-loop philosophy is that a bot shouldn't front students until the eval passes. I also scoped a deployment/hosting path (static course-site widget → HTTPS → a scale-to-zero backend calling a hosted LLM), costed per-student-per-year, but that's future work.

## What this project demonstrates

- **End-to-end applied AI** — data pipeline, retrieval, LLM grounding, UI, and a human-in-the-loop feedback system, built and shipped solo.
- **Retrieval evaluation & debugging** — diagnosing *why* the right chunk isn't retrieved, and fixing it at the data layer (the #56 → #2 story).
- **Human-in-the-loop feedback systems** — purpose-built tooling to capture calibrated reviewer judgments on model outputs, categorize failures by root cause, and build a labeled ground-truth corpus (the shape of RLHF / model-evaluation data work).
- **Eval-first thinking** — correct / grounded / safe, iterate on the KB not the model.
- **Domain-driven design** — assessment/rubric nuance encoded so the bot is right for the right reasons.
- **Responsible data handling** — PII scrubbing and privacy-by-design in a FERPA context (see below).
- **Pragmatic engineering** — local-first, dependency-light, no framework lock-in.

## Privacy & data governance

An AI-in-education system lives or dies on data governance, and this one was built so that **no identifying student data ever reaches the bot or any model context.** That was a deliberate, rigorously enforced constraint, not an afterthought. Four layers:

1. **No model training on student data — at all.** This is retrieval-augmented generation, not fine-tuning. Nothing is trained or tuned on student content; the bot *retrieves* from a fixed knowledge base at query time.
2. **The knowledge base is built exclusively from published course content** — the FAQ pages, section maps, and reference docs I authored, already public on the course website. Raw student emails are **never** indexed and are never in the bot's retrieval space.
3. **The offline gap analysis ran only on de-identified data.** Student email is used *only* offline, to discover which questions the FAQ doesn't yet cover. Before any clustering, analysis, or draft generation, the corpus passes through a **dedicated PII-scrubbing stage** that strips identifying information (names, contact details, and other personal identifiers). Every downstream step operates on the scrubbed corpus — the identifying data is removed at the front of the pipeline, by design.
4. **This repository is a sanitized case study.** Architecture, decisions, and illustrative data-free code only — no emails, no student data, no derived clusters, no feedback logs. It was authored in a separate directory from the working system; the `.gitignore` is a defense-in-depth backstop on top of that separation.

The net effect: the bot answers from public course material, and the one place student text is touched (offline gap-finding) sees only PII-scrubbed data. FERPA-protected information never enters the model's context, the retrieval index, or this repository.

## Tech stack

Python · [Ollama](https://ollama.com) (`nomic-embed-text` embeddings, `qwen2.5:14b` generation) · `numpy` · `gradio` · `pyyaml` · `pdfplumber` · Cloudflare Tunnel (private faculty sharing)

---

*Questions about the design or the eval methodology are welcome — reach me at drjimashe@proton.me.*
