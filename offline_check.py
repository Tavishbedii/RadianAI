"""Radian AI - offline fallback used when Gemini is unavailable.

Combines Radian AI's own components into one result in the same format
the app uses for Gemini results:
  1. claim_cleaner.py  - turns the message into a clear, checkable claim
  2. verify_claim.py   - news + Wikipedia evidence checked with an NLI model
  3. predict.py        - the trained writing-style model (extra signal)
  4. simple rules      - warning signs such as "share now" or ALL CAPS
"""
import re

URGENCY = [
    (r"\b(share|forward)\b.{0,25}\b(now|fast|quickly|everyone|all|maximum|before)\b", "Pressure to share quickly"),
    (r"\b(urgent|breaking|alert|shocking|must watch|must read)\b", "Sensational or urgent language"),
    (r"\b(before (it|this) (gets )?(deleted|removed|banned))\b", "Claims it will be deleted or hidden"),
    (r"\b(100% (true|real|confirmed)|confirmed news|100 percent)\b", "Insists it is true without a source"),
    (r"\b(free|win|winner|prize|lottery|cash ?back)\b.{0,40}\b(click|link|register|claim)\b", "Too-good-to-be-true offer with a link"),
    (r"\b(bit\.ly|tinyurl\.com|goo\.gl|t\.co)/\S+", "Shortened or hidden link"),
]


def red_flags(text):
    flags = [label for pattern, label in URGENCY if re.search(pattern, text, re.IGNORECASE)]
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 30 and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        flags.append("Written mostly in capital letters")
    if text.count("!") >= 3:
        flags.append("Excessive exclamation marks")
    return flags


def style_signal(text):
    """Result from the trained style model, or None if it isn't available."""
    try:
        from predict import predict
        return predict(text)
    except Exception:
        return None


def offline_check(text):
    from claim_cleaner import clean_claims
    from verify_claim import verify          # loads the NLI model on first use

    text = (text or "").strip()
    if not text:
        raise ValueError("The offline verifier needs readable text.")

    claims, method = clean_claims(text)
    claim = claims[0] if claims else text[:300]
    verdict, explanation, used = verify(claim)

    assessment = {"TRUE": "Credible", "FALSE": "Likely Fake"}.get(verdict, "Suspicious")
    evidence_for = [e["text"] for e in used if e.get("entailment", 0) >= 0.7]
    evidence_against = [e["text"] for e in used if e.get("contradiction", 0) >= 0.7]

    confidence = None
    if verdict == "TRUE" and used:
        confidence = sum(e["entailment"] for e in used) / len(used)
    elif verdict == "FALSE" and used:
        confidence = sum(e["contradiction"] for e in used) / len(used)

    sources, seen = [], set()
    for e in used:
        if e["source"] not in seen:
            seen.add(e["source"])
            sources.append({"title": f'{e["type"]}: {e["text"][:90]}', "url": e["source"]})

    flags = red_flags(text)
    reasoning = (f'Gemini was unavailable, so Radian AI used its offline verifier. It searched recent '
                 f'news and Wikipedia for "{claim}" and checked each source with a language-understanding '
                 f'(NLI) model. {explanation}')

    style = style_signal(text)
    if style:
        style_verdict, style_reason = style
        reasoning += f" Our trained writing-style model says: {style_verdict} ({style_reason})."
        if style_verdict == "Likely fake":
            flags.append("Writing style resembles known fake news")

    extra = claims[1:]
    missing = ("This message also contains other claims that were not checked: " + "; ".join(extra)
               if extra else "")

    advice = {
        "Credible": "Multiple sources support this, but open the links to confirm the details.",
        "Likely Fake": "Don't forward this. Sources contradict it.",
        "Suspicious": "Don't forward this until a reliable source confirms it.",
    }[assessment]
    reply = {
        "Credible": "I checked this and news sources do report it.",
        "Likely Fake": "I checked this and news sources contradict it, so please don't forward it.",
        "Suspicious": "I couldn't confirm this from any reliable source, so it's best not to forward it yet.",
    }[assessment]

    return {
        "claim": claim,
        "assessment": assessment,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "missing_context": missing,
        "red_flags": flags,
        "advice": advice,
        "reply_message": reply,
        "sources": sources,
        "engine": f"Offline verifier (claims cleaned with {method})",
    }