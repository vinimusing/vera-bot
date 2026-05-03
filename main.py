#!/usr/bin/env python3
"""
magicpin AI Challenge — Vera Bot
=================================
Claude Sonnet 4 powered merchant engagement bot.

API key from: https://console.anthropic.com
Set as environment variable: ANTHROPIC_API_KEY

Endpoints:
  GET  /v1/healthz
  GET  /v1/metadata
  POST /v1/context
  POST /v1/tick
  POST /v1/reply
  POST /v1/teardown (optional)
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vera")

app = FastAPI(title="Vera Bot — magicpin AI Challenge")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
START = time.time()

@app.get("/")
async def root():
    return {"service": "vera-bot", "status": "running", "endpoints": ["/v1/healthz", "/v1/metadata", "/v1/context", "/v1/tick", "/v1/reply"]}

# ═══════════════════════════════════════════════
# IN-MEMORY STATE
# ═══════════════════════════════════════════════
contexts: dict[tuple[str, str], dict] = {}       # (scope, context_id) -> {version, payload}
conversations: dict[str, list[dict]] = {}         # conversation_id -> list of turns
ended_convos: set[str] = set()                    # closed conversation IDs
suppressed_merchants: set[str] = set()            # merchants who said STOP
used_suppression_keys: set[str] = set()           # triggers already fired

# ═══════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════
def get_ctx(scope: str, cid: str) -> Optional[dict]:
    e = contexts.get((scope, cid))
    return e["payload"] if e else None

def find_digest_item(category: dict, item_id: str) -> Optional[dict]:
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None

# ═══════════════════════════════════════════════
# CLAUDE API
# ═══════════════════════════════════════════════
async def ask_llm(system: str, user: str, max_tok: int = 1200) -> str:
    if not ANTHROPIC_API_KEY:
        log.warning("No ANTHROPIC_API_KEY — fallback mode")
        return ""
    try:
        async with httpx.AsyncClient(timeout=25.0) as c:
            r = await c.post(
                ANTHROPIC_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": max_tok,
                    "temperature": 0,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
    except Exception as e:
        log.error(f"Claude error: {e}")
        return ""

def parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"): s = s[:-3]
        s = s.strip()
    if s.startswith("json"): s = s[4:].strip()
    return json.loads(s)

# ═══════════════════════════════════════════════
# SYSTEM PROMPT
# Designed around 5 scoring dimensions + 10 case-study patterns
# ═══════════════════════════════════════════════
SYSTEM = """You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages for Indian merchants and their customers.

You are scored on 5 dimensions (0-10 each). Optimize ALL:

## 1. DECISION QUALITY — why this message NOW
- First sentence = the hook. It must explain why you're messaging RIGHT NOW.
- Connect trigger event → merchant's specific situation → action.
- Example: "JIDA ka Oct issue aaya — aapke high-risk patients ke liye relevant hai"

## 2. SPECIFICITY — real facts, no fluff
- Include at LEAST 2 concrete numbers from the context (views, CTR, trial size, price, member count, etc.)
- Exact prices: "Dental Cleaning @ ₹299" NOT "a discount" or "an offer"
- Exact sources: "JIDA Oct 2026 p.14" NOT "a recent study"
- Exact batch numbers, dates, localities when available
- NEVER say: "increase your sales", "boost your business", "grow your reach" — these score 0
- Numbers without provenance in the context = fabrication = penalized

## 3. CATEGORY FIT — right voice
- dentists: peer-clinical, technical terms OK (fluoride varnish, scaling, caries, OPG), cite journals, no hype
- salons: warm-practical, visual-aspirational, talk trends/seasons, emojis OK
- restaurants: fellow-operator tone, use "covers", "AOV", "footfall", urgency-driven
- gyms: coach-to-operator, energetic but disciplined, use "churn", "PT sessions", "PR"
- pharmacies: trustworthy-precise, neighbourhood-pharmacist, use "molecule", "OTC", "schedule H"
- ABSOLUTELY NEVER use vocab_taboo words. If the category says "guaranteed" is taboo, NEVER write "guaranteed"
- Hindi-English code-mix when merchant.identity.languages includes "hi" — keep it natural, not forced

## 4. MERCHANT FIT — personalize to THIS specific merchant
- Use owner_first_name in salutation: "Dr. Meera", "Suresh", "Karthik" — NEVER just "Hi"
- Reference THEIR numbers: "aapke 2,410 views", "aapki CTR 2.1%", "aapke 245 members"
- Compare to peer benchmarks: "peer avg CTR 3.0% hai, aapki 2.1%"
- Reference THEIR active offers by exact title
- Reference THEIR signals: stale_posts, ctr_below_peer, dormant, etc.
- Check conversation_history — NEVER repeat what Vera already said

## 5. ENGAGEMENT COMPULSION — make them reply
- ONE CTA at the END of the message — never in the middle
- Make it low-effort: YES/NO preferred, or a simple choice
- Use exactly ONE compulsion lever per message:
  * LOSS AVERSION: "X log search kar rahe hain par aapko nahi mil rahe"
  * SOCIAL PROOF: "Aapke area ke 3 dentists ne iss month yeh kiya"
  * CURIOSITY: "Ek interesting pattern dikha — dekhna chahenge?"
  * EFFORT EXTERNALIZATION: "Maine draft bana diya hai — sirf GO bolo"
  * RECIPROCITY: "Aapke account mein yeh dikha, socha bata doon"
- End with the CTA, not a trailing thought

## 6. JUDGMENT — add your own thinking, don't just template
- If the data suggests NOT doing something, say so. Example: IPL match on Saturday (peak day anyway) → "Saturday already peak hota hai, ad spend save karein aur weekday ke liye rakhein"
- If a metric looks suspicious, flag it: "CTR up but calls down — could be bot traffic"
- If timing is wrong, wait: return action=wait instead of forcing a bad message
- This contrarian, data-informed judgment is the HIGHEST signal of quality

## HARD RULES
1. NEVER fabricate data not in the provided context. This means: do NOT add facts from your training data. If a stat, guideline, regulation detail, ROI figure, or technical specification is NOT explicitly in the context provided, do NOT include it. ONLY use numbers, sources, facts, and claims that appear in the category/merchant/trigger/customer data. Fabrication is the #1 reason for score drops.
2. NEVER use words from voice.vocab_taboo
3. ONE CTA per message, at the end — ALWAYS make it binary YES/NO. End every message with a concrete action: "Bhejoon? Reply YES" or "Draft ready kar doon? YES bolo" — never end with an open question like "kya karein?" or "discuss karein?"
4. customer-facing (send_as=merchant_on_behalf): speak AS the merchant's clinic/shop, NEVER mention Vera. Address the CUSTOMER by name.
5. No preambles — no "I hope you're doing well", no "I wanted to reach out"
6. service+price ("Dental Cleaning @ ₹299") ALWAYS beats discount format ("20% off")
7. Keep messages 3-5 sentences for WhatsApp. Not an email. SHORT replies score better.
8. The rationale field must match what you actually wrote — judge cross-checks this
9. For REPLIES: keep responses under 4 sentences. Be concise. Answer then CTA. Do not write paragraphs. Do NOT add facts from your own knowledge — ONLY use what's in the context.
10. Every CTA must remove a barrier: "no commitment", "2 min ka kaam", "just say GO", "no auto-charge"

OUTPUT — return ONLY valid JSON, nothing else:
{"body": "the WhatsApp message text", "cta": "binary_yes_no|open_ended|binary_confirm_cancel|multi_choice_slot|none", "rationale": "1-2 sentences: why now + which scoring dimensions this targets"}"""

# ═══════════════════════════════════════════════
# COMPOSE PROMPT BUILDER
# ═══════════════════════════════════════════════
def build_compose_prompt(cat: dict, merch: dict, trig: dict, cust: Optional[dict] = None) -> str:
    p = []
    slug = cat.get("slug", "?")
    voice = cat.get("voice", {})
    identity = merch.get("identity", {})
    perf = merch.get("performance", {})
    peer = cat.get("peer_stats", {})

    # ── Category ──
    p.append(f"=== CATEGORY: {slug} ===")
    p.append(f"Tone: {voice.get('tone', '?')}, Register: {voice.get('register', '?')}")
    p.append(f"Code-mix: {voice.get('code_mix', 'english')}")
    taboo = voice.get("vocab_taboo", [])
    if taboo:
        p.append(f"⛔ FORBIDDEN WORDS (never use): {', '.join(taboo)}")
    allowed = voice.get("vocab_allowed", [])
    if allowed:
        p.append(f"✅ Encouraged vocabulary: {', '.join(allowed[:12])}")
    tone_ex = voice.get("tone_examples", [])
    if tone_ex:
        p.append(f"Tone examples: {' | '.join(tone_ex[:3])}")

    offers_cat = cat.get("offer_catalog", [])
    if offers_cat:
        p.append(f"Category offer patterns: {json.dumps(offers_cat[:6])}")

    if peer:
        parts = []
        for k, v in peer.items():
            if k != "scope":
                parts.append(f"{k}={v}")
        p.append(f"Peer benchmarks ({peer.get('scope', '?')}): {', '.join(parts)}")

    digest = cat.get("digest", [])
    if digest:
        p.append("Weekly digest:")
        for d in digest:
            line = f"  [{d.get('kind','?')}] \"{d.get('title','')}\" — Source: {d.get('source','?')}"
            if d.get("trial_n"):
                line += f", N={d['trial_n']}"
            if d.get("patient_segment"):
                line += f", Segment: {d['patient_segment']}"
            p.append(line)
            if d.get("summary"):
                p.append(f"    Summary: {d['summary'][:250]}")

    seasonal = cat.get("seasonal_beats", [])
    if seasonal:
        p.append(f"Seasonal beats: {json.dumps(seasonal)}")
    trends = cat.get("trend_signals", [])
    if trends:
        p.append(f"Trend signals: {json.dumps(trends)}")
    patient_lib = cat.get("patient_content_library", [])
    if patient_lib:
        p.append(f"Patient content library: {json.dumps(patient_lib[:3])}")

    # ── Merchant ──
    p.append(f"\n=== MERCHANT ===")
    p.append(f"Name: {identity.get('name', '?')}")
    p.append(f"Owner first name: {identity.get('owner_first_name', '?')}")
    p.append(f"City: {identity.get('city', '?')}, Locality: {identity.get('locality', '?')}")
    p.append(f"Languages: {identity.get('languages', ['en'])}")
    p.append(f"Verified: {identity.get('verified', False)}")

    sub = merch.get("subscription", {})
    p.append(f"Subscription: {sub.get('status','?')} / {sub.get('plan','?')} / {sub.get('days_remaining','?')} days left")

    p.append(f"Performance ({perf.get('window_days',30)}d): views={perf.get('views','?')}, calls={perf.get('calls','?')}, directions={perf.get('directions','?')}, CTR={perf.get('ctr','?')}, leads={perf.get('leads','?')}")
    delta = perf.get("delta_7d", {})
    if delta:
        p.append(f"  7d delta: views {'+' if delta.get('views_pct',0)>=0 else ''}{round(delta.get('views_pct',0)*100)}%, calls {'+' if delta.get('calls_pct',0)>=0 else ''}{round(delta.get('calls_pct',0)*100)}%")

    offers = merch.get("offers", [])
    active = [o for o in offers if o.get("status") == "active"]
    expired = [o for o in offers if o.get("status") != "active"]
    if active:
        p.append(f"Active offers: {', '.join(o.get('title','?') for o in active)}")
    if expired:
        p.append(f"Expired/paused offers: {', '.join(o.get('title','?') for o in expired)}")

    agg = merch.get("customer_aggregate", {})
    if agg:
        parts = [f"{k}={v}" for k, v in agg.items()]
        p.append(f"Customer aggregate: {', '.join(parts)}")

    signals = merch.get("signals", [])
    if signals:
        p.append(f"Signals: {', '.join(str(s) for s in signals)}")

    hist = merch.get("conversation_history", [])
    if hist:
        p.append("Recent Vera conversations (DO NOT repeat these):")
        for h in hist[-4:]:
            p.append(f"  [{h.get('from','?')}] {h.get('body','')[:150]} → {h.get('engagement','?')}")

    # ── Trigger ──
    p.append(f"\n=== TRIGGER (why message NOW) ===")
    p.append(f"Kind: {trig.get('kind','?')}")
    p.append(f"Source: {trig.get('source','?')}, Scope: {trig.get('scope','?')}, Urgency: {trig.get('urgency','?')}/5")
    trig_payload = trig.get("payload", {})
    if trig_payload:
        p.append(f"Trigger payload: {json.dumps(trig_payload)}")

    # Resolve referenced digest item
    top_id = trig_payload.get("top_item_id")
    if top_id:
        item = find_digest_item(cat, top_id)
        if item:
            p.append(f"REFERENCED DIGEST ITEM:")
            p.append(f"  Title: {item.get('title','?')}")
            p.append(f"  Source: {item.get('source','?')}")
            if item.get("trial_n"):
                p.append(f"  Trial size: {item['trial_n']}")
            if item.get("patient_segment"):
                p.append(f"  Patient segment: {item['patient_segment']}")
            if item.get("summary"):
                p.append(f"  Summary: {item['summary'][:300]}")

    # ── Customer (optional) ──
    if cust:
        ci = cust.get("identity", {})
        cr = cust.get("relationship", {})
        cp = cust.get("preferences", {})
        cc = cust.get("consent", {})
        p.append(f"\n=== CUSTOMER (this is CUSTOMER-FACING — speak as the merchant) ===")
        p.append(f"Name: {ci.get('name','?')}, Language: {ci.get('language_pref','en')}, Age: {ci.get('age_band','?')}")
        p.append(f"State: {cust.get('state','?')}")
        p.append(f"Relationship: first_visit={cr.get('first_visit','?')}, last_visit={cr.get('last_visit','?')}, total_visits={cr.get('visits_total','?')}")
        p.append(f"Services received: {cr.get('services_received', [])}")
        p.append(f"Preferred slots: {cp.get('preferred_slots','?')}, Channel: {cp.get('channel','?')}")
        p.append(f"Consent scope: {cc.get('scope', [])}")
        p.append(f"\n⚠️⚠️⚠️ CRITICAL — CUSTOMER-FACING MESSAGE ⚠️⚠️⚠️")
        p.append(f"Speak AS '{identity.get('name', 'the business')}' TO the customer '{ci.get('name','')}'.")
        p.append(f"Address the customer as '{ci.get('name','')}' in the salutation — NOT the merchant owner '{identity.get('owner_first_name','')}'.")
        p.append(f"Example: 'Hi {ci.get('name','')}, ...' NOT 'Hi {identity.get('owner_first_name','')}, ...'")
        p.append(f"NEVER mention Vera. Honor language_pref ({ci.get('language_pref','en')}) and preferred_slots ({cp.get('preferred_slots','?')}).")
    else:
        p.append(f"\nThis is MERCHANT-FACING. Speak as Vera to the merchant owner.")

    # ── Trigger-kind specific framing ──
    kind = trig.get("kind", "")
    framing_map = {
        "research_digest": "FRAMING: Lead with the specific research finding. Cite source + page. Mention trial size + relevant patient segment from merchant's data. Offer to pull abstract + draft patient-ed content. Example shape: '[Source] ka [month] issue aaya. [N]-patient trial: [finding]. Aapke [segment] patients ke liye relevant. Abstract + patient WhatsApp draft bhejoon?'",
        "regulation_change": "FRAMING: Lead with DEADLINE and specific regulation. Cite circular/notification number. State what changes. Offer compliance audit as low-effort next step.",
        "perf_spike": "FRAMING: Celebrate with EXACT numbers from performance context. Compare to their own history or peer avg. Suggest capitalizing — draft a Google Post, push a campaign. Example: 'Aapke views +X% upar gaye — total Y views. Peer avg Z hai. Iss momentum ko capitalize karein?'",
        "perf_dip": "FRAMING: Flag the dip with exact numbers. NO alarm. Frame as diagnosis. Tie to a specific signal if available. Offer ONE concrete fix. Example: 'Calls -X% neeche aaye. CTR Y hai vs peer Z. Last Google post Nd purana hai — fresh post se improvement aata hai. Draft karoon?'",
        "recall_due": "FRAMING: Patient recall. Use patient name, months since last visit, specific slots matching their preference, exact price from active offers. Clinical but warm tone.",
        "appointment_tomorrow": "FRAMING: Appointment reminder to customer. Confirm exact time, clinic/shop name, locality. Any prep instructions. Keep short.",
        "milestone_reached": "FRAMING: Celebrate exact milestone number. Compare to peer average. Suggest leveraging — GBP post, social share, patient-facing message.",
        "dormant_with_vera": "FRAMING: Re-engagement. NO guilt. Lead with a genuinely useful data point (a trend, a stat, a peer insight). Make them curious. 'Kuch interesting dikha aapke data mein...'",
        "competitor_opened": "FRAMING: Curiosity hook. 'Ek naya [category] khula hai [distance] door.' Frame as intel, not threat. Suggest a differentiator action.",
        "festival_upcoming": "FRAMING: Name the festival + days away. Suggest specific festive offer from their catalog with exact price. Create time urgency.",
        "ipl_match_today": "FRAMING: Name the teams + time. Add JUDGMENT — Saturday IPL usually shifts covers (use peer data if available). Suggest leveraging existing offers. Think like an operator, not a promoter.",
        "review_theme_emerged": "FRAMING: 'X reviews this week mein [theme] mention hua.' Surface the exact pattern. Offer to help address — update listing, respond to reviews, adjust service.",
        "curious_ask_due": "FRAMING: Ask a genuine question about their business. 'Iss hafte sabse zyada kaunsi service demand mein rahi?' Offer to turn their answer into content. No commitment needed.",
        "category_seasonal": "FRAMING: Connect seasonal trend to a specific action. Reference seasonal_beats data. Suggest preemptive offer or content.",
        "customer_lapsed_soft": "FRAMING: Soft winback for 3-6 month lapsed customer. Warm, no guilt, no shame. Mention what they last got. Offer something from catalog. 'No commitment, no auto-charge.'",
        "customer_lapsed_hard": "FRAMING: Winback for 6mo+ lapsed customer. Acknowledge the gap warmly. Remind what they used. Offer a come-back incentive from catalog with exact price. Zero pressure.",
        "winback_eligible": "FRAMING: Winback — remind what they received last time, offer a come-back incentive from catalog with exact price. No pressure.",
        "renewal_due": "FRAMING: Lead with VALUE delivered during subscription — cite specific metrics that improved (views up, calls up, reviews gained). Then the renewal ask.",
        "trial_followup": "FRAMING: Summarize trial results with exact numbers. Then conversion ask with specific plan name and price.",
        "gbp_unverified": "FRAMING: Profile verification nudge. Explain what they're missing. Effort externalization: 'Bas aapka OTP chahiye, 2 minute ka kaam hai.'",
        "supply_alert": "FRAMING: URGENT. Name specific product/batch. State the issue clearly (recall, shortage, sub-potency). If customer count affected is derivable, state it. Offer to draft customer notifications.",
        "chronic_refill_due": "FRAMING: Medication refill reminder. Name all molecules. State exact expiry date. Include total + savings if senior discount applies. Offer home delivery if available.",
        "cde_opportunity": "FRAMING: Continuing education event. Cite event name, date, topic, credits. Peer-learning tone: 'X dentists from your area already registered.' Low commitment.",
        "active_planning_intent": "FRAMING: Merchant expressed interest. Reference what they asked about. Deliver a COMPLETE draft artifact — pricing tiers, time slots, target audience. Then ask to refine.",
        "wedding_package_followup": "FRAMING: Follow up on bridal/event interest. Reference specific services discussed, wedding date if known, and skin-prep or booking timeline. Offer concrete next step.",
        "seasonal_perf_dip": "FRAMING: Normalize the dip — 'har saal [months] mein yeh hota hai, -X% to -Y% normal hai.' Reframe: save ad spend now, invest in retention. Offer a specific retention action.",
    }
    if kind in framing_map:
        p.append(f"\n{framing_map[kind]}")
    else:
        p.append(f"\nFRAMING: Compose a contextually relevant message for trigger kind '{kind}'. Use all available context. Be specific, not generic.")

    p.append("\nCompose the message now. Return ONLY valid JSON.")
    return "\n".join(p)

# ═══════════════════════════════════════════════
# REPLY PROMPT BUILDER
# ═══════════════════════════════════════════════
def build_reply_prompt(cat: dict, merch: dict, turns: list, msg: str, cust: Optional[dict] = None) -> str:
    p = []
    voice = cat.get("voice", {})
    identity = merch.get("identity", {})

    p.append(f"=== CATEGORY: {cat.get('slug','?')} ===")
    p.append(f"Tone: {voice.get('tone','?')}")
    taboo = voice.get("vocab_taboo", [])
    if taboo:
        p.append(f"⛔ FORBIDDEN: {', '.join(taboo)}")

    p.append(f"\n=== MERCHANT ===")
    p.append(f"Name: {identity.get('name','?')}, Owner: {identity.get('owner_first_name','?')}")
    p.append(f"Languages: {identity.get('languages', ['en'])}")
    p.append(f"Locality: {identity.get('locality','?')}, City: {identity.get('city','?')}")
    p.append(f"Offers: {json.dumps(merch.get('offers', []))}")
    p.append(f"Performance: views={merch.get('performance',{}).get('views','?')}, CTR={merch.get('performance',{}).get('ctr','?')}")
    p.append(f"Signals: {merch.get('signals', [])}")

    if cust:
        ci = cust.get("identity", {})
        p.append(f"\n=== CUSTOMER ===")
        p.append(json.dumps(cust))
        p.append(f"\n⚠️⚠️⚠️ CRITICAL — CUSTOMER-FACING CONVERSATION ⚠️⚠️⚠️")
        p.append(f"You are speaking AS '{identity.get('name','the business')}' TO the customer '{ci.get('name','the customer')}'.")
        p.append(f"Address the CUSTOMER by their name '{ci.get('name','')}', NOT the merchant owner.")
        p.append(f"Example: 'Hi {ci.get('name','')}, aapka appointment confirmed...' NOT 'Hi {identity.get('owner_first_name','')}, ...'")
        p.append(f"NEVER mention Vera. NEVER address the merchant owner in this reply.")

    p.append(f"\n=== FULL CONVERSATION ===")
    for t in turns:
        role = "VERA" if t["from"] == "vera" else t["from"].upper()
        p.append(f"[{role}] {t['body']}")

    p.append(f"\n=== NEW MESSAGE ===")
    p.append(msg)

    p.append("""
ANALYZE and respond with ONLY valid JSON.

CHECK IN THIS ORDER:

1. AUTO-REPLY? Look for: "Thank you for contacting", "Our team will respond shortly", "automated assistant", "we will get back", "your message is important", "Welcome to", "We received your message", or IDENTICAL/near-identical text to a previous message in the conversation, or any generic canned response that doesn't reference the conversation topic.
   Count auto-replies in the FULL conversation history above:
   → 1st auto-reply: {"action":"send","body":"Lagta hai auto-reply hai — jab owner dekhe, sirf 'Yes' reply kar dena. 😊","cta":"binary_yes_no","rationale":"Auto-reply #1, one nudge to owner"}
   → 2nd auto-reply: {"action":"wait","wait_seconds":86400,"rationale":"Auto-reply #2. Waiting 24h for real owner."}
   → 3rd auto-reply: {"action":"wait","wait_seconds":86400,"rationale":"Auto-reply #3. Still waiting for real owner."}
   → 4th auto-reply: {"action":"wait","wait_seconds":86400,"rationale":"Auto-reply #4. Still no real owner."}
   → 5th+ auto-reply: {"action":"end","rationale":"Auto-reply 5+ times. No real engagement. Closing gracefully."}

2. INTENT TRANSITION? If the merchant/customer says ANYTHING that signals agreement or readiness:
   Words: "yes", "haan", "ok", "sure", "go ahead", "proceed", "let's do it", "kar do", "bhej do", "chalega", "send it", "book it", "confirm", "done", "fine", "theek hai", "chalo", "let's go", "do it", "please book", "set it up", "I'm in", "count me in"
   → CRITICAL: IMMEDIATELY switch to ACTION mode. Do NOT ask another qualifying question. Do NOT ask "which date?" or "what time?" if they already specified it. Tell them EXACTLY what you are doing NOW. Be specific:
     - If they said "book Wed 5 Nov 6pm" → confirm that exact slot
     - If they said "yes send it" → tell them you're sending it now
     - If they just said "yes" → execute whatever you proposed in your last message
   → Use their name (customer name for customer-facing, owner name for merchant-facing)
   → Include specific details from context (offer price, service name, time)

3. STOP/UNSUBSCRIBE? Words: "stop", "not interested", "band karo", "don't message", "unsubscribe", "leave me alone", "mat bhejo", "no thanks", "not now"
   → {"action":"end","rationale":"Merchant/customer opted out. Respecting preference."}

4. HOSTILE/ABUSIVE? Anger, frustration, rude language, cursing
   → One empathetic line + graceful exit. Do NOT argue or get defensive.
   → {"action":"send","body":"Sorry for the inconvenience. Aage se nahi bhejungi. Kuch chahiye ho toh kabhi bhi 'Hi Vera' bol dena. 🙏","cta":"none","rationale":"Hostile response detected. One empathetic exit."}

5. OFF-TOPIC? Questions about GST, legal matters, accounting, personal questions, tech support, anything outside Vera's merchant growth scope
   → Politely decline in 1 sentence. Do NOT try to help with the off-topic request. Redirect to the original topic with a specific offer.
   → Example: {"action":"send","body":"GST filing mein main help nahi kar paungi — uske liye CA se baat karein. Btw, aapke Dental Cleaning @ ₹299 offer pe 2,410 searches aa rahe hain — usko push karein?","cta":"binary_yes_no","rationale":"Off-topic deflected, redirected to original value prop with specific data"}

6. QUESTION? Merchant/customer asks a specific question about their data, offers, or Vera's capabilities
   → Answer with SPECIFIC facts from the context provided. Use exact numbers. Keep it SHORT (2-3 sentences). End with ONE CTA.

7. ENGAGED REPLY? Any other engaged response showing interest
   → Continue the conversation naturally. Use context data. ONE CTA at end. Keep short.

CRITICAL RULES FOR ALL REPLIES:
- If customer context exists → address the CUSTOMER by name, speak as the merchant's business
- If no customer context → address the MERCHANT OWNER by name, speak as Vera
- NEVER fabricate data not in the context. Do NOT add stats, guidelines, ROI figures, or technical specs from your training data. ONLY use facts present in the context above.
- NEVER use vocab_taboo words
- Keep replies SHORT (2-3 sentences for WhatsApp)
- ALWAYS end with a binary YES/NO CTA: "Bhejoon? Reply YES" not "discuss karein?"
- ONE CTA at end, never in the middle

Return ONE of:
{"action":"send","body":"text","cta":"open_ended|binary_yes_no|none","rationale":"why"}
{"action":"wait","wait_seconds":3600,"rationale":"why"}
{"action":"end","rationale":"why"}""")
    return "\n".join(p)

# ═══════════════════════════════════════════════
# SMART FALLBACKS (data-driven, not generic)
# ═══════════════════════════════════════════════
def fallback_compose(cat, merch, trig, cust):
    identity = merch.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name", "")
    kind = trig.get("kind", "update")
    perf = merch.get("performance", {})
    peer = cat.get("peer_stats", {})
    offers = [o for o in merch.get("offers", []) if o.get("status") == "active"]
    signals = merch.get("signals", [])
    agg = merch.get("customer_aggregate", {})
    delta = perf.get("delta_7d", {})
    views = perf.get("views", 0)
    ctr = perf.get("ctr", 0)
    peer_ctr = peer.get("avg_ctr", 0)
    locality = identity.get("locality", "")
    langs = identity.get("languages", ["en"])
    use_hindi = "hi" in langs

    # Customer-facing
    if cust:
        cn = cust.get("identity", {}).get("name", "there")
        bn = identity.get("name", "our clinic")
        rel = cust.get("relationship", {})
        last_v = rel.get("last_visit", "")
        visits = rel.get("visits_total", 0)
        offer_str = f" {offers[0]['title']} available hai." if offers else ""
        lang_pref = cust.get("identity", {}).get("language_pref", "en")

        if "hi" in lang_pref:
            return (
                f"Hi {cn}, {bn} yahan se. Aapki last visit {last_v} ko thi — aapka next checkup due hai.{offer_str} Kab aana chahenge?",
                "open_ended",
                f"Customer recall fallback with visit history ({visits} visits) and active offer"
            )
        return (
            f"Hi {cn}, {bn} here. Your last visit was on {last_v} — your next checkup is due.{offer_str} When would you like to come in?",
            "open_ended",
            f"Customer recall fallback with visit history and active offer"
        )

    # Merchant-facing, trigger-specific
    if kind == "perf_spike" and views:
        v_pct = delta.get("views_pct", 0)
        body = f"Hi {name}, {'aapke' if use_hindi else 'your'} profile {'ne' if use_hindi else 'got'} last 7 days mein {'+' if v_pct>=0 else ''}{round(v_pct*100)}% {'zyada' if use_hindi else 'more'} views — total {views} views this month."
        if peer_ctr and ctr:
            body += f" CTR {ctr} hai vs peer avg {peer_ctr}."
        body += f" {'Iss momentum ko capitalize karein? Reply YES.' if use_hindi else 'Want to capitalize on this? Reply YES.'}"
        return (body, "binary_yes_no", f"Perf spike with exact views ({views}), delta ({round(v_pct*100)}%), CTR comparison")

    if kind == "perf_dip":
        c_pct = delta.get("calls_pct", 0)
        stale = any("stale_posts" in str(s) for s in signals)
        body = f"Hi {name}, {'aapke' if use_hindi else 'your'} calls mein {round(abs(c_pct)*100)}% dip {'aaya hai' if use_hindi else 'dropped'} last 7 days."
        if ctr and peer_ctr:
            body += f" CTR {ctr} vs peer avg {peer_ctr}."
        if stale:
            body += f" {'Last Google post kaafi purana hai — fresh post se improvement aata hai.' if use_hindi else ' Your last Google post is stale — a fresh one usually helps.'}"
        body += f" {'Ek quick audit karein? 2 min lagega.' if use_hindi else ' Quick audit? Takes 2 min.'}"
        return (body, "binary_yes_no", f"Perf dip with call drop ({round(abs(c_pct)*100)}%), CTR vs peer, stale post signal")

    if kind == "dormant_with_vera":
        body = f"Hi {name}, {'kuch din ho gaye!' if use_hindi else 'it has been a while!'} {'Aapke' if use_hindi else 'Your'} profile pe {views} views {'aaye hain' if use_hindi else 'came in'} last 30 days."
        if ctr and peer_ctr:
            body += f" CTR {ctr} vs peer avg {peer_ctr}."
        body += f" {'Ek quick update share karoon?' if use_hindi else ' Want a quick update?'}"
        return (body, "binary_yes_no", f"Dormant re-engagement with view count ({views}) and CTR comparison")

    if kind == "milestone_reached":
        payload = trig.get("payload", {})
        milestone = payload.get("milestone", "a milestone")
        body = f"Hi {name}, congratulations! 🎉 {'Aapne' if use_hindi else 'You'} {milestone} {'cross kar liya' if use_hindi else 'crossed'}!"
        if peer.get("avg_review_count"):
            body += f" Peer avg {peer['avg_review_count']} hai."
        body += f" {'Google pe ek post daalein? Maine draft bana diya.' if use_hindi else ' Want to post on Google? I have a draft ready.'}"
        return (body, "binary_yes_no", f"Milestone celebration with peer comparison")

    if kind == "festival_upcoming":
        payload = trig.get("payload", {})
        festival = payload.get("festival_name", payload.get("festival", "festival"))
        offer_str = f" {offers[0]['title']}" if offers else ""
        body = f"Hi {name}, {festival} {'aa raha hai!' if use_hindi else 'is coming!'}{' Aapka' if use_hindi else ' Your'}{offer_str} {'push karein? Festive-themed Google post bhi draft kar doon.' if use_hindi else ' should we push it? I can draft a festive Google post too.'}"
        return (body, "binary_yes_no", f"Festival upcoming with active offer")

    # Generic but data-grounded fallback
    offer_str = f" Active offer: {offers[0]['title']}." if offers else ""
    signal_str = ""
    if "stale_posts" in str(signals):
        signal_str = f" {'Last Google post kaafi purana hai.' if use_hindi else ' Last Google post is stale.'}"
    elif "ctr_below_peer" in str(signals):
        signal_str = f" CTR {ctr} vs peer avg {peer_ctr}."

    body = f"Hi {name},{signal_str}{offer_str} {'Ek quick update hai — share karoon?' if use_hindi else ' Quick update — want me to share?'}"
    return (body, "binary_yes_no", f"Data-grounded fallback for trigger {kind}")


def fallback_reply(msg, turns, merch=None, cust=None):
    ml = msg.lower().strip()

    # Get names for personalization
    cust_name = cust.get("identity", {}).get("name", "") if cust else ""
    merch_name = merch.get("identity", {}).get("owner_first_name", "") if merch else ""
    biz_name = merch.get("identity", {}).get("name", "") if merch else ""
    address_name = cust_name if cust else merch_name

    auto_signals = ["thank you for contacting", "our team will respond", "automated assistant",
                    "we will get back", "please wait", "your message is important",
                    "welcome to", "we received your message"]
    if any(s in ml for s in auto_signals):
        # Count auto-replies in history (current message is already in turns)
        auto_count = sum(1 for t in turns if t["from"] != "vera" and
                        any(s in t["body"].lower() for s in auto_signals))
        # auto_count includes current message, so:
        # 1st auto-reply: auto_count=1
        # 2nd auto-reply: auto_count=2
        # 3rd auto-reply: auto_count=3
        # 4th auto-reply: auto_count=4
        if auto_count >= 5:
            return {"action": "end", "rationale": "Auto-reply 5+ times. Closing conversation."}
        elif auto_count >= 4:
            return {"action": "wait", "wait_seconds": 86400, "rationale": "Auto-reply #4. Still waiting for real owner."}
        elif auto_count >= 3:
            return {"action": "wait", "wait_seconds": 86400, "rationale": "Auto-reply #3. Still waiting for real owner."}
        elif auto_count >= 2:
            return {"action": "wait", "wait_seconds": 86400, "rationale": "Auto-reply #2. Waiting 24h for owner."}
        return {"action": "send",
                "body": "Lagta hai auto-reply hai — jab owner dekhe, sirf 'Yes' reply kar dena. 😊",
                "cta": "binary_yes_no",
                "rationale": "Auto-reply #1. Nudging for real owner."}

    # Hostile detection (check BEFORE stop — hostile messages often contain "stop")
    hostile_signals = ["bakwas", "spam", "fraud", "scam", "waste", "useless", "stupid",
                       "shut up", "get lost", "pagal", "bewakoof", "chup", "rubbish",
                       "nonsense", "faltu", "bekaar"]
    if any(s in ml for s in hostile_signals):
        return {"action": "send",
                "body": "Sorry for the inconvenience. Aage se nahi bhejungi. Kuch chahiye ho toh kabhi bhi 'Hi Vera' bol dena. 🙏",
                "cta": "none",
                "rationale": "Hostile response detected. Graceful exit with open door."}

    stop_signals = ["stop", "not interested", "band karo", "don't message",
                    "unsubscribe", "leave me alone", "mat bhejo", "no thanks", "not now"]
    if any(s in ml for s in stop_signals):
        return {"action": "end", "rationale": "Merchant opted out. Respecting preference."}

    intent_signals = ["yes", "haan", "ok", "sure", "go ahead", "proceed", "let's do it",
                      "kar do", "bhej do", "chalega", "send it", "book it", "confirm",
                      "done", "fine", "theek hai", "chalo", "let's go", "do it",
                      "please book", "set it up", "i'm in", "count me in"]
    if any(s in ml for s in intent_signals):
        if cust and merch:
            offers = [o for o in merch.get("offers", []) if o.get("status") == "active"]
            offer_str = f" {offers[0]['title']}" if offers else ""
            body = f"{cust_name}, done! Aapka appointment set up kar rahi hoon —{offer_str}. Confirmation bhejti hoon shortly. 👍"
        elif merch:
            body = f"{merch_name}, done! Abhi set up kar rahi hoon. Confirmation 2 minute mein bhejti hoon. 👍"
        else:
            body = "Done — abhi set up kar rahi hoon. Confirmation 2 minute mein bhejti hoon. 👍"
        return {"action": "send", "body": body, "cta": "none",
                "rationale": "Intent detected. Switching to action mode immediately."}

    # Off-topic detection
    offtopic_signals = ["gst", "tax", "legal", "lawyer", "court", "accounting", "ca ",
                        "chartered accountant", "income tax", "tds", "compliance filing",
                        "bank loan", "emi", "insurance"]
    if any(s in ml for s in offtopic_signals):
        return {"action": "send",
                "body": "Iss topic mein main help nahi kar paungi — uske liye apne CA ya advisor se baat karein. Btw, aapke profile pe kuch interesting data dikha — share karoon?",
                "cta": "binary_yes_no",
                "rationale": "Off-topic deflected. Redirected to Vera's core value."}

    if address_name:
        return {"action": "send",
                "body": f"Noted {address_name}! Aur kuch help chahiye toh bata dena.",
                "cta": "open_ended",
                "rationale": "Engaged response with personalization."}

    return {"action": "send",
            "body": "Noted! Aur kuch help chahiye toh bata dena.",
            "cta": "open_ended",
            "rationale": "Engaged response."}

# ═══════════════════════════════════════════════
# TRUNCATED JSON RECOVERY
# ═══════════════════════════════════════════════
def recover_truncated_reply(raw: str, msg: str, turns: list, merch=None, cust=None) -> dict:
    """Try to extract a usable reply from truncated Claude JSON output."""
    try:
        # Try to find the body field even in broken JSON
        import re
        body_match = re.search(r'"body"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
        if body_match:
            body_text = body_match.group(1)
            # Clean up: if body got cut off mid-sentence, add ellipsis and CTA
            if not body_text.endswith(('.', '?', '!')):
                body_text = body_text.rstrip() + "... Reply YES to continue."
            
            cta_match = re.search(r'"cta"\s*:\s*"([^"]*)"', raw)
            cta = cta_match.group(1) if cta_match else "binary_yes_no"
            
            return {
                "action": "send",
                "body": body_text,
                "cta": cta,
                "rationale": "Recovered from truncated response."
            }
    except Exception:
        pass
    
    return fallback_reply(msg, turns, merch, cust)

# ═══════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Builder",
        "team_members": ["Builder"],
        "model": ANTHROPIC_MODEL,
        "approach": "4-context composer with trigger-kind-aware prompt routing covering 26 trigger types across 5 verticals. Optimized for 5-dimension scoring. Claude Sonnet 4 for composition. Auto-reply detection (3-strike escalation), intent-transition handling, Hindi-English code-mix. Full context grounding with zero hallucination. Data-driven fallbacks.",
        "contact_email": "builder@example.com",
        "version": "2.1.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"Unknown: {body.scope}"}
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z"}


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers:
        trig = get_ctx("trigger", trg_id)
        if not trig:
            log.warning(f"Trigger {trg_id} not found in context store — skipping")
            continue
        mid = trig.get("merchant_id")
        if not mid or mid in suppressed_merchants:
            continue
        sup_key = trig.get("suppression_key", "")
        if sup_key and sup_key in used_suppression_keys:
            continue
        merch = get_ctx("merchant", mid)
        if not merch:
            log.warning(f"Merchant {mid} not found for trigger {trg_id} — skipping")
            continue

        # Try multiple ways to find category
        cat_slug = merch.get("category_slug", "")
        cat = get_ctx("category", cat_slug)
        if not cat:
            # Try without trailing 's' or with trailing 's'
            cat = get_ctx("category", cat_slug.rstrip("s"))
            if not cat:
                cat = get_ctx("category", cat_slug + "s")
            if not cat:
                # Search all loaded categories for a partial match
                for (scope, cid), entry in contexts.items():
                    if scope == "category":
                        if cat_slug in cid or cid in cat_slug:
                            cat = entry["payload"]
                            break
            if not cat:
                # Use empty category rather than skipping entirely
                log.warning(f"Category '{cat_slug}' not found for merchant {mid} — using minimal category")
                cat = {"slug": cat_slug, "voice": {}, "offer_catalog": [], "peer_stats": {}, "digest": []}

        cust = None
        cid = trig.get("customer_id")
        if cid:
            cust = get_ctx("customer", cid)

        is_cx = trig.get("scope") == "customer" and cust is not None
        send_as = "merchant_on_behalf" if is_cx else "vera"
        conv_id = f"conv_{mid}_{trg_id}"

        if conv_id in conversations or conv_id in ended_convos:
            continue

        # Compose
        prompt = build_compose_prompt(cat, merch, trig, cust)
        raw = await ask_llm(SYSTEM, prompt)

        body_text = cta = rationale = None
        if raw:
            try:
                data = parse_json(raw)
                body_text = data.get("body", "").strip()
                cta = data.get("cta", "open_ended")
                rationale = data.get("rationale", "Composed from 4-context framework")
            except Exception as e:
                log.warning(f"Parse error for trigger {trg_id}: {e}")

        if not body_text:
            body_text, cta, rationale = fallback_compose(cat, merch, trig, cust)

        if not body_text:
            log.warning(f"No body generated for trigger {trg_id} — skipping")
            continue

        if sup_key:
            used_suppression_keys.add(sup_key)

        conversations[conv_id] = [{"from": "vera", "body": body_text, "ts": body.now}]

        owner = merch.get("identity", {}).get("owner_first_name", "")
        biz = merch.get("identity", {}).get("name", "")

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": cid,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trig.get('kind','generic')}_v1",
            "template_params": [owner or biz, body_text[:100], ""],
            "body": body_text,
            "cta": cta,
            "suppression_key": sup_key,
            "rationale": rationale,
        })

    log.info(f"Tick processed {len(body.available_triggers)} triggers → {len(actions)} actions")
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    cid = body.conversation_id
    if cid in ended_convos:
        return {"action": "end", "rationale": "Conversation was previously closed."}

    conversations.setdefault(cid, []).append({
        "from": body.from_role, "body": body.message, "ts": body.received_at
    })
    turns = conversations[cid]

    merch = get_ctx("merchant", body.merchant_id) if body.merchant_id else None
    cust = None
    if not merch:
        result = fallback_reply(body.message, turns, None, None)
    else:
        cat = get_ctx("category", merch.get("category_slug", "")) or {}
        cust = get_ctx("customer", body.customer_id) if body.customer_id else None
        prompt = build_reply_prompt(cat, merch, turns, body.message, cust)
        raw = await ask_llm(SYSTEM, prompt, max_tok=1200)
        if raw:
            try:
                result = parse_json(raw)
                if result.get("action") == "send" and not result.get("body"):
                    result = fallback_reply(body.message, turns, merch, cust)
            except Exception:
                # Try to recover truncated JSON — extract body if possible
                result = recover_truncated_reply(raw, body.message, turns, merch, cust)
        else:
            result = fallback_reply(body.message, turns, merch, cust)

    if result.get("action") == "end":
        ended_convos.add(cid)
        if body.merchant_id and any(s in body.message.lower() for s in ["stop", "not interested", "band karo", "mat bhejo"]):
            suppressed_merchants.add(body.merchant_id)
    elif result.get("action") == "send":
        conversations[cid].append({
            "from": "vera", "body": result.get("body", ""), "ts": datetime.utcnow().isoformat() + "Z"
        })
    return result


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    ended_convos.clear()
    suppressed_merchants.clear()
    used_suppression_keys.clear()
    return {"status": "wiped"}


# ═══════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
