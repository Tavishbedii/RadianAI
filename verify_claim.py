"""Radian AI - claim cleaner.

Turns messy user input into clear, checkable claims BEFORE verification:
  "nana partekar is alive i saw him on juhu beach yesterday"  ->  "Nana Patekar is alive"
  "is diwali on 8 nov??"                                       ->  "Diwali 2026 is on November 8"
  "bhai cyclone aa raha hai kya andhra mein"                   ->  "A cyclone is approaching Andhra Pradesh"

Uses Gemini when an API key and quota are available (fixes spelling, translates
Hindi/Hinglish, splits multiple claims). Falls back to basic rules otherwise.
"""
import json
import os
import re
from datetime import date

try:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    load_dotenv()
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

CLEANER_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_CLAIMS = 3

PROMPT = """Today's date is {today}.
A user sent this message to a fake news checker:

"{message}"

Rewrite it as at most {max_claims} clear, standalone factual claims that can be checked against news.
Rules:
- Write each claim in English, as a statement (never a question).
- Fix spelling of names and places.
- Translate Hindi or Hinglish into English.
- Remove personal stories, opinions and requests (e.g. "I saw him", "please share").
- Replace relative dates like "yesterday" or "this year" with real dates where possible.
- If the message contains no checkable factual claim, return an empty list.
Respond with ONLY a JSON list of strings, for example: ["Claim one", "Claim two"]"""

AUX_VERBS = {"is", "are", "was", "were", "will", "can"}
DROP_AUX = {"do", "does", "did", "has", "have", "had"}
PREDICATE_HINTS = {"on", "in", "at", "alive", "dead", "coming", "approaching", "going",
                   "from", "by", "not", "still", "really", "true", "real", "fake",
                   "closed", "open", "banned", "available", "free", "safe"}
FILLER_START = r"^(bhai|bro|guys|friends|omg|wow|breaking|urgent|alert|hey|hi|hello|plz|please)\b[\s,!:.-]*"
PERSONAL = (r"\b(i|we)\s+(saw|seen|heard|think|believe|met|read|watched|noticed|feel|guess)\b.*$"
            r"|\b(my|our)\s+(friend|uncle|aunt|cousin|brother|sister|neighbou?r)\b.*$"
            r"|\b(please|plz|pls)\s+(share|forward)\b.*$"
            r"|\bshare\s+(this|it|with)\b.*$|\bforward\s+(this|it|to)\b.*$")


def _question_to_statement(sentence):
    words = sentence.split()
    if len(words) < 3:
        return sentence
    first = words[0].lower()
    if first in DROP_AUX:
        return " ".join(words[1:])
    if first in AUX_VERBS:
        for i in range(2, len(words)):
            if words[i].lower() in PREDICATE_HINTS:
                return " ".join(words[1:i] + [first] + words[i:])
        return " ".join([words[1], first] + words[2:])
    return sentence


def rule_based_clean(message):
    """Basic offline cleaning: no spelling fixes or translation."""
    text = re.sub(r"\s+", " ", message).strip()
    claims = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        part = part.strip()
        for _ in range(3):
            part = re.sub(FILLER_START, "", part, flags=re.IGNORECASE).strip()
        part = re.sub(PERSONAL, "", part, flags=re.IGNORECASE).strip(" ,.!?-:")
        part = _question_to_statement(part.rstrip("?! ").strip())
        if len(part.split()) >= 2:
            claims.append(part[0].upper() + part[1:])
    return claims[:MAX_CLAIMS]


def gemini_clean(message):
    client = genai.Client()   # reads GEMINI_API_KEY / GOOGLE_API_KEY from .env
    response = client.models.generate_content(
        model=CLEANER_MODEL,
        contents=PROMPT.format(today=date.today().isoformat(),
                               message=message.replace('"', "'"),
                               max_claims=MAX_CLAIMS),
        config=types.GenerateContentConfig(temperature=0,
                                           response_mime_type="application/json"),
    )
    text = re.sub(r"```(json)?", "", response.text or "").strip()
    claims = json.loads(text)
    return [str(c).strip() for c in claims if str(c).strip()][:MAX_CLAIMS]


def clean_claims(message):
    """Return (claims, method). method is "Gemini" or "basic rules"."""
    if GEMINI_AVAILABLE:
        try:
            return gemini_clean(message), "Gemini"
        except Exception:
            pass   # no key, quota reached, or bad output -> fall back
    return rule_based_clean(message), "basic rules"


if __name__ == "__main__":
    while True:
        msg = input("Message: ").strip()
        if not msg:
            break
        claims, method = clean_claims(msg)
        print(f"  [{method}] {claims}\n")