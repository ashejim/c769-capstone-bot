"""
Illustrative excerpt: the retrieval core of the capstone FAQ bot.

This is a sanitized, self-contained slice of the production engine —
enough to show the design and code style, with no data dependencies.
The real system wires this into a Gradio chat UI and a feedback-triage
console; all three frontends call the same UI-agnostic `BotEngine`.

Design notes worth calling out:
  - The embedding matrix is pre-normalized ONCE at load, so each query
    is a single matrix-vector product (cosine similarity) — no per-query
    normalization of the corpus, no external vector DB needed at this
    scale (a few hundred chunks).
  - A hard similarity floor short-circuits to a human hand-off WITHOUT
    calling the LLM: a safety choice (don't answer from thin retrieval)
    and a latency choice (skip generation on clearly off-topic queries).
  - The LLM is grounded strictly on retrieved chunks; the prompt forbids
    answering beyond them and requires source citation.

Embeddings/generation run locally via Ollama; no cloud provider is in
the runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Retrieval tuning. Loose lower bound lets the LLM see weak matches too
# (it is instructed to refuse if the context is too thin); the hard
# floor below is where we don't even bother calling the model.
TOP_K = 5
HARD_FLOOR_SIM = 0.40   # below this, hand off to a human, no LLM call
GOOD_MATCH_SIM = 0.55   # below this, flag the answer as possibly incomplete


@dataclass
class Chunk:
    """One indexable retrieval unit (FAQ entry, section-map block, or
    a reference-doc section)."""
    id: str
    source_file: str
    heading: str
    phase: str | None       # e.g. "Task 1" — used in the embed-text prefix
    text: str
    embedding: np.ndarray    # 1-D vector


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def retrieve(
    query_emb: np.ndarray,
    chunks: list[Chunk],
    matrix: np.ndarray,
    *,
    top_k: int = TOP_K,
) -> list[tuple[Chunk, float]]:
    """Rank chunks by cosine similarity. `matrix` is pre-normalized, so
    similarity is one matrix-vector multiply against the normalized query."""
    q = normalize(query_emb)
    sims = matrix @ q
    idx = np.argsort(-sims)[:top_k]
    return [(chunks[i], float(sims[i])) for i in idx]


class Retriever:
    """Loads the index once; retrieves per query. The production engine
    wraps this with the LLM grounding call, citation rendering, link
    resolution, and the hand-off framing."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # Pre-normalize the corpus once. The local embedder does not emit
        # unit-length vectors, so normalizing each row up front makes every
        # subsequent query a single dot-product.
        matrix = np.stack([c.embedding for c in chunks])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0          # guard against any zero vectors
        self.matrix = matrix / norms

    def top_matches(self, query_emb: np.ndarray) -> list[tuple[Chunk, float]]:
        return retrieve(query_emb, self.chunks, self.matrix)

    def should_hand_off(self, results: list[tuple[Chunk, float]]) -> bool:
        """True when the best match is below the hard floor — the engine
        returns a human hand-off and never calls the LLM."""
        best_sim = results[0][1] if results else 0.0
        return best_sim < HARD_FLOOR_SIM


# The grounding prompt (abridged) — the guardrail that keeps the bot
# honest. Full version adds the per-task section-letter disambiguation
# and the automated-assistant framing.
SYSTEM_PROMPT = """\
You are a first-layer triage assistant for a university IT capstone course.

- Answer ONLY from the retrieved context provided, or from what the
  student told you earlier in the conversation. If neither covers the
  question, say so and recommend the student contact their instructor.
- Cite the source(s) you used, by file name, in a "Sources:" list.
- If the current message is too vague to answer confidently, ask ONE
  specific clarifying question instead of guessing.
- A rubric section letter means different things by task (e.g. Task 2
  section H = "Outcome"; Task 3 section H = "Conclusion"). If the task
  isn't clear, ask before answering.
- You are an automated assistant, not the instructor. Don't invent
  policies, deadlines, or details that aren't in the retrieved context.
"""
