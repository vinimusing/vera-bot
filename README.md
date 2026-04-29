# Vera Bot — magicpin AI Challenge

## Approach

Gemini 2.0 Flash powered 4-context composer with trigger-kind-aware prompt routing, optimized for all 5 scoring dimensions.

**Architecture:** FastAPI server → versioned in-memory context store → trigger-aware prompt builder → Gemini API (structured JSON) → response validation + data-driven fallback

**Key decisions:**

1. **5-dimension-optimized prompting** — System prompt explicitly maps to the judge's rubric: decision quality (why now), specificity (real numbers only), category fit (voice/vocab enforcement), merchant fit (personalization), engagement compulsion (single CTA with compulsion lever)

2. **26 trigger-type coverage** — Every trigger kind (research_digest, perf_spike, recall_due, supply_alert, etc.) gets specific framing instructions so messages match the situation naturally rather than using a one-size-fits-all template

3. **Context grounding with zero hallucination** — The prompt explicitly forbids fabrication. Numbers, sources, offers, and signals are only used if they exist in the pushed context. The rationale field cross-checks against the actual output

4. **3-strike auto-reply escalation** — WhatsApp Business auto-replies are detected via keyword matching + conversation history counting. 1st: nudge owner. 2nd: wait 24h. 3rd+: end gracefully

5. **Data-driven fallbacks** — When LLM is unavailable, fallback messages still use real merchant data (view counts, CTR vs peer benchmarks, active offer prices, signals) — never generic text

6. **Hindi-English code-mix** — Natural code-mixing when merchant languages include "hi", matching real Indian merchant communication patterns

**Tradeoffs:**
- Gemini 2.0 Flash over larger models: faster response within 30s budget, free tier, sufficient quality for structured composition
- In-memory storage: simple, sufficient for
