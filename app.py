import io, os, re, json, html, uuid, base64, difflib, datetime, importlib.util
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# On Streamlit Community Cloud the settings come from the app's Secrets, not a .env file.
# Copy them into the environment so the rest of the app (and the Gemini client) can read them.
try:
    for _key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "USE_GOOGLE_SEARCH", "OFFLINE_FALLBACK", "WHISPER_MODEL"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass   # no secrets file (normal on a laptop that uses .env)
from google import genai
from google.genai import types

# =====================================================================
# Settings
# =====================================================================
MODEL = "gemini-3.5-flash-lite"   # works on the free tier (checked with check_key.py)
USE_GOOGLE_SEARCH = os.getenv("USE_GOOGLE_SEARCH", "false").lower() == "true"    # True needs a Gemini plan that includes Google Search grounding
OFFLINE_FALLBACK = os.getenv("OFFLINE_FALLBACK", "true").lower() == "true"    # use Radian AI's own verifier when Gemini fails
TESSERACT_PATH = os.getenv("TESSERACT_PATH") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe" if os.name == "nt" else "tesseract")
# The offline verifier needs PyTorch + Transformers. The online (lite) deployment leaves them out.
OFFLINE_AVAILABLE = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
GEMINI_VIDEO_LIMIT_MB = 18   # larger videos are sent to Gemini as extracted text only
# Speech-to-text (Whisper) is only on the full local install. Without it (the online version),
# videos must be small enough for Gemini to watch and listen to directly.
WHISPER_AVAILABLE = importlib.util.find_spec("faster_whisper") is not None
VIDEO_TYPES = ["mp4", "mov", "webm", "m4v", "mkv"] if WHISPER_AVAILABLE else ["mp4", "mov", "webm"]
VIDEO_MAX_MB = None if WHISPER_AVAILABLE else GEMINI_VIDEO_LIMIT_MB
# File extension -> the video format name Gemini accepts (browsers call .mov "video/quicktime").
GEMINI_VIDEO_MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/mov", ".webm": "video/webm"}

TEAM = [("Shubham", "Backend & bot"), ("Siddharth", "OCR"),
        ("Tavish", "AI model"), ("Swayam", "Frontend")]

st.set_page_config(page_title="Radian AI", page_icon="🔎", layout="wide",
                   initial_sidebar_state="auto")

# Presentation only: the labels below are exactly the ones the backend returns.
VERDICT = {
    "Credible":    {"color": "#3FB27F", "symbol": "✓", "badge": "green",  "icon": ":material/check_circle:",
                    "meaning": "Reliable sources report the same thing."},
    "Suspicious":  {"color": "#E0A43C", "symbol": "!", "badge": "orange", "icon": ":material/help:",
                    "meaning": "This couldn't be confirmed. Don't forward it until you've checked further."},
    "Likely Fake": {"color": "#E2585B", "symbol": "✕", "badge": "red",    "icon": ":material/cancel:",
                    "meaning": "Reliable sources contradict it, or it shows strong signs of misinformation."},
}

# =====================================================================
# Session state (all stored for this browser session only)
# =====================================================================
DEFAULTS = {
    "history": [],          # list of result dicts, newest first
    "page": "verify",       # verify | history | about
    "img": None,            # the screenshot being checked: {bytes, name, type, size, fid}
    "img_invalid": None,    # name of a file that couldn't be read as an image
    "up_key": 0,            # bumping this resets the upload widget
    "up_seen": None,        # last file seen in the upload widget
    "img_result_id": None,  # id of the result shown in the screenshot tab
    "vid": None,            # the video being checked: {bytes, name, type, size, fid}
    "vid_up_key": 0,        # bumping this resets the video upload widget
    "vid_seen": None,       # last video seen in the upload widget
    "vid_result_id": None,  # id of the result shown in the video tab
    "txt_result_id": None,  # id of the result shown in the message tab
    "msg_saved": "",        # message text, kept when switching pages
    "msg_invalid": False,
    "job": None,            # a verification waiting to run: {"mode": "img"|"vid"|"txt", "fresh": bool}
    "errors": {},           # last error per mode
    "open_id": None,        # history item opened in full
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, (list, dict)) else type(v)()

# =====================================================================
# Styling
# =====================================================================
st.markdown("""
<style>
:root {
  --bg: #0B0D12; --side: #11141B; --card: #151922; --raised: #1B202B;
  --text: #F3F4F6; --muted: #9CA3AF; --line: #262C38; --accent: #7C8CFF;
}
.block-container { padding-top: 2.2rem; max-width: 1280px; }

/* Sidebar */
.brand-name { font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em; margin: 0; }
.brand-tag { color: var(--muted); font-size: .92rem; margin: .1rem 0 1rem 0; }
section[data-testid="stSidebar"] .stButton button { justify-content: flex-start; padding-left: .8rem; padding-right: .8rem; }
section[data-testid="stSidebar"] .stButton button > div { justify-content: flex-start; width: 100%; }
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"]:hover { background: rgba(255,255,255,.05); }
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
  background: rgba(124,140,255,.16); color: var(--text); border: 1px solid rgba(124,140,255,.45);
}
.side-label { color: var(--muted); font-size: .85rem; font-weight: 600; margin: 1.1rem 0 .4rem 0; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; }
.stats div { background: var(--raised); border-radius: 8px; padding: .5rem .7rem; }
.stats b { display: block; font-size: 1.2rem; }
.stats span { color: var(--muted); font-size: .82rem; }

/* Header */
.app-header { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem 1rem; }
.app-title { font-size: clamp(1.8rem, 3vw, 2.3rem); font-weight: 700; letter-spacing: -.02em; margin: 0; }
.ai-badge { font-size: .72rem; font-weight: 600; letter-spacing: .06em; color: var(--accent);
            border: 1px solid rgba(124,140,255,.45); border-radius: 999px; padding: .2rem .65rem; }
.app-sub { color: var(--muted); font-size: 1.02rem; margin: .35rem 0 1.2rem 0; max-width: 70ch; }

/* Step indicator */
.steps { display: flex; gap: .45rem; list-style: none; padding: 0 0 .2rem 0; margin: .2rem 0 1.3rem 0;
         overflow-x: auto; scrollbar-width: thin; }
.steps li { display: flex; align-items: center; gap: .5rem; white-space: nowrap; flex: 0 0 auto;
            padding: .42rem .8rem; border: 1px solid var(--line); border-radius: 8px;
            color: var(--muted); font-size: .9rem; margin: 0; }
.steps .num { font-weight: 600; font-variant-numeric: tabular-nums; }
.steps li.done { color: var(--text); }
.steps li.done .num { color: #3FB27F; }
.steps li.current { color: var(--text); border-color: var(--accent); background: rgba(124,140,255,.12); }
.steps li.current .num { color: var(--accent); }
.steps li.skipped { border-style: dashed; opacity: .65; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }

/* Inputs */
.panel-title { font-size: 1.1rem; font-weight: 600; margin: 0 0 .15rem 0; }
.panel-help { color: var(--muted); font-size: .92rem; margin: 0 0 .7rem 0; }
.preview { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: .5rem; margin-top: .6rem; }
.preview img { display: block; width: 100%; max-height: 380px; object-fit: contain; border-radius: 6px; }
.preview p { color: var(--muted); font-size: .85rem; margin: .45rem .2rem .1rem; overflow-wrap: anywhere; }
.empty-state { border: 1px dashed var(--line); border-radius: 10px; padding: 2.2rem 1.4rem; text-align: center; color: var(--muted); }
.empty-state b { color: var(--text); display: block; font-size: 1.02rem; margin-bottom: .25rem; }

/* Verdict */
.verdict-card { background: var(--card); border: 1px solid var(--line); border-left: 5px solid var(--c);
                border-radius: 10px; padding: 1.15rem 1.3rem; margin-bottom: .4rem; }
.v-label { display: flex; align-items: center; gap: .7rem; font-size: 1.7rem; font-weight: 700; letter-spacing: -.01em; }
.v-icon { width: 2rem; height: 2rem; border-radius: 50%; background: var(--c); color: #0B0D12;
          display: grid; place-items: center; font-size: 1.05rem; font-weight: 700; flex: 0 0 auto; }
.v-meaning { margin: .5rem 0 .9rem 0; font-size: 1rem; }
.meter { height: 8px; background: rgba(255,255,255,.08); border-radius: 99px; overflow: hidden; max-width: 420px; }
.meter span { display: block; height: 100%; background: var(--c); border-radius: 99px; }
.v-conf { color: var(--muted); font-size: .9rem; margin: .45rem 0 0 0; }
.v-conf b { color: var(--text); }

.section-title { font-size: 1.05rem; font-weight: 600; margin: 1.4rem 0 .45rem 0; }
.section-note { color: var(--muted); font-size: .86rem; margin: -.25rem 0 .6rem 0; }
.claim { background: var(--raised); border-radius: 8px; padding: .8rem 1rem; font-size: 1.02rem; line-height: 1.55; }
.ev-group { margin: 0 0 .8rem 0; }
.ev-group h4 { font-size: .95rem; font-weight: 600; margin: 0 0 .3rem 0; padding: 0; }
.ev-group ul { margin: 0; padding-left: 1.2rem; }
.ev-group li { margin-bottom: .25rem; line-height: 1.5; }
.flag { display: inline-block; padding: .22rem .7rem; margin: 0 .35rem .4rem 0; border-radius: 999px;
        border: 1px solid rgba(226,88,91,.5); background: rgba(226,88,91,.1); font-size: .88rem; }

.src-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: .6rem; }
.src-card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: .8rem .95rem; min-width: 0; }
.src-card .pub { font-weight: 600; overflow-wrap: anywhere; margin-bottom: .35rem; }
.src-card .n { color: var(--muted); font-size: .8rem; margin-bottom: .15rem; }
.src-card a { color: var(--accent); font-size: .9rem; text-decoration: none; }
.src-card a:hover { text-decoration: underline; }

.meta { display: flex; flex-wrap: wrap; gap: .4rem 1.6rem; margin: 0; color: var(--muted); font-size: .86rem; }
.meta dt { display: inline; } .meta dd { display: inline; margin: 0 0 0 .35rem; color: var(--text); }

a:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { .meter span { transition: none; } }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Backend (unchanged logic: OCR, Gemini with Google Search, caching)
# =====================================================================
@st.cache_resource
def get_client():
    return genai.Client()   # reads GEMINI_API_KEY from .env


def run_ocr(image_bytes):
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((img.width * 2, img.height * 2))
        text = pytesseract.image_to_string(img, lang="eng+hin")
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


# Extended with three optional fields (evidence_for, evidence_against, missing_context).
# Older fields are unchanged, so earlier results still display correctly.
PROMPT = """You are a careful fact-checking assistant for people in India who receive news and forwards on WhatsApp and social media.
Input type: {source}
Extracted text (may contain OCR errors): {text}

Tasks:
1. Identify the single main factual claim.
2. Use Google Search to check it against reliable sources (major news outlets, government sites, and fact-checkers such as PIB Fact Check, Alt News and BOOM).
3. Look for signs of misinformation: urgency ("share now"), no named source, sensational language, edited visuals, too-good-to-be-true offers, suspicious links.

Respond ONLY with JSON, no markdown, in exactly this format:
{"claim": "the main claim in one sentence",
 "assessment": "Credible" or "Suspicious" or "Likely Fake",
 "confidence": a number between 0 and 1,
 "reasoning": "2-3 plain sentences explaining the verdict",
 "evidence_for": ["what reliable sources say that supports the claim"],
 "evidence_against": ["what reliable sources say that contradicts the claim"],
 "missing_context": "important context the original leaves out, or an empty string",
 "red_flags": ["short phrase", "short phrase"],
 "advice": "one sentence telling the reader what to do",
 "reply_message": "a short, polite message (max 2 sentences) the user can send back to the group, in the same language as the original"}
Only include evidence you actually found in your search. Use empty lists or an empty string when there is nothing.
If the claim cannot be verified, use "Suspicious"."""


def parse_json(raw):
    raw = re.sub(r"```(json)?", "", raw or "")
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


def recent_news(text, from_media=False):
    """Recent Google News headlines about the claim, found by Radian AI's own search (free, no API key)."""
    try:
        from news_search import fetch_news
        from claim_cleaner import rule_based_clean
        from offline_check import strip_ocr_noise, best_claim
        cleaned = strip_ocr_noise(text) if from_media else text
        claim = best_claim(rule_based_clean(cleaned), cleaned)
        return fetch_news(claim, max_results=6)
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def ask_gemini(image_bytes, mime, text, news=None):
    parts = []
    if image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
    prompt = (PROMPT.replace("{source}", "video" if (mime or "").startswith("video")
                                     else "screenshot" if image_bytes else "pasted message")
                    .replace("{text}", text or "(none)"))
    prompt += f"\nToday's date is {datetime.date.today().isoformat()}."
    if not USE_GOOGLE_SEARCH:
        prompt = prompt.replace("Use Google Search to check it against", "Check it using what you know about")
        if news:
            headlines = "\n".join(f"- {n['title']} ({n['date']})" for n in news)
            prompt = prompt.replace("Only include evidence you actually found in your search.",
                                    "Only include evidence you actually found in the headlines below or are sure of.")
            prompt += ("\nYou cannot search the web yourself, but Radian AI searched Google News and found these "
                       "recent headlines. They may be newer than your own knowledge, so treat them as evidence "
                       "when they are about the same claim, and ignore unrelated ones:\n" + headlines)
        else:
            prompt += ("\nYou cannot search the web for this check. Only state evidence you are confident about, "
                       "and say so if the claim may be too recent for you to know.")
    parts.append(prompt)

    response = get_client().models.generate_content(
        model=MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())] if USE_GOOGLE_SEARCH else None,
            temperature=0.2,
        ),
    )
    result = parse_json(response.text)

    sources = []
    meta = response.candidates[0].grounding_metadata
    if meta and meta.grounding_chunks:
        for chunk in meta.grounding_chunks[:5]:
            if chunk.web:
                sources.append({"title": chunk.web.title, "url": chunk.web.uri})
    if not sources and news:
        sources = [{"title": n["title"], "url": n["link"]} for n in news[:5]]
        result["source_kind"] = "news"
    result["sources"] = sources
    return result


def _str_list(value):
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def clean(r):
    a = str(r.get("assessment", "")).strip().lower()
    r["assessment"] = "Likely Fake" if "fake" in a else "Credible" if "credible" in a else "Suspicious"
    # Confidence is only shown when the model actually supplies a usable number.
    try:
        c = float(r.get("confidence"))
        c = c / 100 if c > 1 else c
        r["confidence"] = max(0.0, min(1.0, c))
    except (TypeError, ValueError):
        r["confidence"] = None
    for key in ("red_flags", "evidence_for", "evidence_against"):
        r[key] = _str_list(r.get(key))
    r["missing_context"] = str(r.get("missing_context") or "").strip()
    r.setdefault("claim", "Claim could not be identified.")
    r.setdefault("reasoning", "")
    r.setdefault("advice", "")
    r.setdefault("reply_message", "")
    r["sources"] = [s for s in r.get("sources", []) if s.get("url", "").startswith(("http://", "https://"))]
    return r


class UserFacingError(ValueError):
    """An error whose message is already written for the user."""


def friendly_error(e):
    msg = str(e)
    if isinstance(e, UserFacingError) or "offline verifier" in msg:
        return msg
    if "API key" in msg or "401" in msg or "403" in msg or "PERMISSION" in msg:
        return ("The Gemini API key is missing or not valid. Check GEMINI_API_KEY in your .env file "
                "(or in the app's Secrets when deployed), then restart the app.")
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "The free-tier limit was reached. Wait a minute and try again."
    if "404" in msg or "not found" in msg.lower():
        return "The AI model name wasn't found. Update MODEL at the top of app.py with the current name from AI Studio."
    if isinstance(e, json.JSONDecodeError) or "Expecting" in msg:
        return "The AI's answer couldn't be read. Try again."
    if "connect" in msg.lower() or "timed out" in msg.lower():
        return "Couldn't reach the AI service. Check your internet connection and try again."
    return "The verification couldn't be completed. Try again, or turn on technical details in the sidebar to see why."


# =====================================================================
# Helpers
# =====================================================================
def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def find_similar(text, claim, exclude_id=None):
    """Find an earlier check of (nearly) the same text or claim in this session."""
    for item in st.session_state.history:
        if item["id"] == exclude_id:
            continue
        for a, b in ((text, item.get("input_text")), (claim, item.get("claim"))):
            a, b = normalize(a), normalize(b)
            if len(a) > 20 and len(b) > 20 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.9:
                return item
    return None


def get_item(rid):
    return next((h for h in st.session_state.history if h["id"] == rid), None)


def pct(r):
    return None if r.get("confidence") is None else round(r["confidence"] * 100)


def make_summary(r):
    conf = f" ({pct(r)}% model-reported confidence)" if pct(r) is not None else ""
    lines = [f"Radian AI verdict: {r['assessment']}{conf}", f"Claim: {r['claim']}"]
    if r["reasoning"]:
        lines.append(f"Why: {r['reasoning']}")
    if r["sources"]:
        lines.append("Sources: " + ", ".join(s["url"] for s in r["sources"][:3]))
    lines.append(f"Checked {r['time']}. AI-assisted result, not a final verdict.")
    return "\n".join(lines)


def make_report(r):
    conf = f" ({pct(r)}% model-reported confidence)" if pct(r) is not None else ""
    lines = [
        "RADIAN AI VERIFICATION REPORT",
        f"Verification ID: {r['id']}",
        f"Checked: {r['time']}",
        f"Input: {r['input_type']}",
        "",
        f"Verdict: {r['assessment']}{conf}",
        f"Claim: {r['claim']}",
        "",
        f"Reasoning: {r['reasoning']}",
    ]
    if r["evidence_for"]:
        lines += ["", "Evidence supporting the claim:"] + [f"- {x}" for x in r["evidence_for"]]
    if r["evidence_against"]:
        lines += ["", "Evidence contradicting the claim:"] + [f"- {x}" for x in r["evidence_against"]]
    if r["missing_context"]:
        lines += ["", f"Missing context: {r['missing_context']}"]
    if r["red_flags"]:
        lines += ["", "Warning signs:"] + [f"- {f}" for f in r["red_flags"]]
    if r["advice"]:
        lines += ["", f"What to do: {r['advice']}"]
    if r["sources"]:
        lines += ["", "Sources:"] + [f"- {s['title']}: {s['url']}" for s in r["sources"]]
    if r.get("input_text"):
        lines += ["", "Original text:", r["input_text"]]
    lines += ["", "AI-assisted verification aid, not a final verdict."]
    return "\n".join(lines)


# =====================================================================
# Actions (button callbacks)
# =====================================================================
def go(page):
    st.session_state.page = page
    st.session_state.open_id = None
    st.session_state.up_seen = None      # upload widget resets on page change; keep the saved image
    st.session_state.vid_seen = None     # same for the saved video


def queue(mode, fresh=False):
    st.session_state.job = {"mode": mode, "fresh": fresh}
    st.session_state.errors.pop(mode, None)


def queue_message():
    text = st.session_state.get("msg_input", "")
    st.session_state.msg_saved = text
    if len(text.strip()) < 10:
        st.session_state.msg_invalid = True
    else:
        st.session_state.msg_invalid = False
        queue("txt")


def save_message():
    st.session_state.msg_saved = st.session_state.get("msg_input", "")
    if st.session_state.msg_invalid and len(st.session_state.msg_saved.strip()) >= 10:
        st.session_state.msg_invalid = False


def clear_message():
    st.session_state.msg_input = ""
    st.session_state.msg_saved = ""
    st.session_state.msg_invalid = False
    st.session_state.txt_result_id = None
    st.session_state.errors.pop("txt", None)


def remove_image():
    st.session_state.img = None
    st.session_state.img_invalid = None
    st.session_state.img_result_id = None
    st.session_state.up_seen = None
    st.session_state.up_key += 1
    st.session_state.errors.pop("img", None)


def remove_video():
    st.session_state.vid = None
    st.session_state.vid_result_id = None
    st.session_state.vid_seen = None
    st.session_state.vid_up_key += 1
    st.session_state.errors.pop("vid", None)


def open_result(rid):
    st.session_state.page = "history"
    st.session_state.open_id = rid
    st.session_state.up_seen = None
    st.session_state.vid_seen = None


def set_feedback(rid, value):
    item = get_item(rid)
    if item:
        item["feedback"] = value
    st.toast("Thanks, your feedback was saved.")


def clear_history():
    st.session_state.history = []
    st.session_state.img_result_id = None
    st.session_state.txt_result_id = None
    st.session_state.vid_result_id = None
    st.session_state.open_id = None


# =====================================================================
# Verification pipeline: the stages shown are the real stages being run
# =====================================================================
def run_pipeline(mode):
    image = st.session_state.img if mode == "img" else None
    video = st.session_state.vid if mode == "vid" else None
    text = st.session_state.msg_saved.strip() if mode == "txt" else ""
    offline_text = None

    with st.status("Analyzing the claim and checking available evidence...", expanded=True) as status:
        if image:
            if st.session_state.get("use_ocr", True):
                st.write("Extracting text from the screenshot (OCR)")
                text = run_ocr(image["bytes"])
                st.write(f"Extracted {len(text.split())} words" if text
                         else "OCR found no readable text, so the AI will read the image directly")
            else:
                st.write("OCR is turned off, so the AI will read the image directly")
        if video:
            st.write("Reading the video: speech to text (Whisper) and on-screen text (OCR on frames). "
                     "The first run downloads the speech model and can take a few minutes.")
            try:
                from video_reader import read_video
                info = read_video(video["bytes"], video["name"], st.session_state.get("use_ocr", True))
            except Exception as e:
                status.update(label="Verification failed", state="error", expanded=False)
                return None, e
            text = info["combined"]
            offline_text = ". ".join(x for x in (info["on_screen"], info["transcript"]) if x)
            for note in info["notes"]:
                st.write(note)
            st.write(f"Transcribed {len(info['transcript'].split())} spoken words and found "
                     f"{len(info['on_screen'].split())} words of on-screen text")
        r, gemini_error = None, None
        if st.session_state.get("offline_only"):
            st.write("Offline mode is on, so Gemini is skipped")
        else:
            st.write("Identifying the claim and searching reliable sources")
            try:
                media_bytes, media_type = None, None
                if image:
                    media_bytes, media_type = image["bytes"], image["type"]
                elif video and video["size"] <= GEMINI_VIDEO_LIMIT_MB * 1024 * 1024:
                    vtype = GEMINI_VIDEO_MIME.get(os.path.splitext(video["name"])[1].lower())
                    if vtype:
                        media_bytes, media_type = video["bytes"], vtype
                if video and not media_bytes and not text:
                    raise UserFacingError(
                        "No speech or on-screen text could be read from this video, and Gemini can only watch "
                        f"MP4, MOV or WEBM videos up to {GEMINI_VIDEO_LIMIT_MB} MB. Upload a shorter MP4 clip.")
                news = []
                if not USE_GOOGLE_SEARCH and text:
                    st.write("Searching recent news about the claim (Google News)")
                    news = recent_news(text, from_media=bool(image or video))
                    st.write(f"Found {len(news)} recent headlines to give the AI as evidence" if news
                             else "No recent headlines found, so the AI will rely on its own knowledge")
                r = clean(dict(ask_gemini(media_bytes, media_type, text, news)))
                r["engine"] = f"Gemini ({MODEL})"
            except Exception as e:
                gemini_error = e
                st.write(f"Gemini unavailable: {friendly_error(e)}")

        if r is None:
            if not OFFLINE_AVAILABLE or not (OFFLINE_FALLBACK or st.session_state.get("offline_only")):
                status.update(label="Verification failed", state="error", expanded=False)
                return None, gemini_error
            check_text = offline_text if video else text
            if not check_text:
                status.update(label="Verification failed", state="error", expanded=False)
                return None, gemini_error or ValueError(
                    "The offline verifier needs readable text. Paste the message text, or turn on OCR "
                    "in Advanced options for screenshots and videos.")
            st.write("Switching to Radian AI's offline verifier: cleaning the claim, searching news and "
                     "Wikipedia, and checking each source with the NLI model "
                     "(the first run loads the model and can take a minute)")
            try:
                from offline_check import offline_check
                r = clean(offline_check(check_text, from_screenshot=bool(image or video)))
            except Exception as e:
                status.update(label="Verification failed", state="error", expanded=False)
                return None, gemini_error or e
        status.update(label=f"Verification complete: {r['assessment']}", state="complete", expanded=False)

    now = datetime.datetime.now()
    rid = "RV-" + uuid.uuid4().hex[:6].upper()
    similar = find_similar(text, r.get("claim"))
    r.update({"id": rid, "time": now.strftime("%d %b %Y, %I:%M %p"),
              "input_text": text, "input_type": "Video" if video else "Screenshot" if image else "Message",
              "used_ocr": bool(image or video) and bool(text), "feedback": None,
              "similar_to": similar["id"] if similar else None})
    st.session_state.history.insert(0, r)
    return r, None


def process_job(mode):
    """Run a queued verification for this mode, then refresh so buttons re-enable."""
    job = st.session_state.job
    if not job or job["mode"] != mode:
        return
    if mode == "img" and not st.session_state.img:
        st.session_state.job = None
        return
    if mode == "vid" and not st.session_state.vid:
        st.session_state.job = None
        return
    try:
        if job["fresh"]:
            ask_gemini.clear()
        r, err = run_pipeline(mode)
        if r:
            st.session_state[f"{mode}_result_id"] = r["id"]
        else:
            st.session_state.errors[mode] = {"message": friendly_error(err),
                                             "detail": f"{type(err).__name__}: {err}"}
    finally:
        st.session_state.job = None
    st.rerun()


# =====================================================================
# UI pieces
# =====================================================================
def step_bar(steps):
    notes = {"done": "completed", "current": "current step", "todo": "not started", "skipped": "not needed"}
    items = []
    for i, (label, state) in enumerate(steps, 1):
        num = "✓" if state == "done" else f"{i:02d}"
        aria = ' aria-current="step"' if state == "current" else ""
        extra = " (not needed)" if state == "skipped" else ""
        items.append(f'<li class="{state}"{aria}><span class="num" aria-hidden="true">{num}</span>'
                     f'{label}{extra}<span class="sr">, step {i}, {notes[state]}</span></li>')
    st.markdown(f'<ol class="steps" aria-label="Verification progress">{"".join(items)}</ol>',
                unsafe_allow_html=True)


def empty_state(title, text):
    st.markdown(f'<div class="empty-state" role="status"><b>{title}</b>{text}</div>', unsafe_allow_html=True)


def show_error(mode):
    err = st.session_state.errors.get(mode)
    if not err:
        return False
    st.error(err["message"], icon=":material/error:")
    st.button("Try again", key=f"retry_{mode}", icon=":material/refresh:",
              on_click=queue, args=(mode,), disabled=st.session_state.job is not None)
    if st.session_state.get("tech"):
        with st.expander("Technical details"):
            st.code(err["detail"], language=None, wrap_lines=True)
    return True


def render_result(r, ctx):
    """ctx is 'img', 'vid', 'txt' or 'hist'. Everything shown comes from the stored result."""
    e = html.escape
    v = VERDICT[r["assessment"]]
    p = pct(r)

    # A. Verdict
    conf_html = ""
    if p is not None:
        conf_html = (f'<div class="meter" role="meter" aria-label="Model-reported confidence" '
                     f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{p}"><span style="width:{p}%"></span></div>'
                     f'<p class="v-conf">Model-reported confidence: <b>{p}%</b></p>')
    st.markdown(
        f'<section class="verdict-card" style="--c:{v["color"]}" role="status" aria-label="Verdict">'
        f'<div class="v-label"><span class="v-icon" aria-hidden="true">{v["symbol"]}</span>{e(r["assessment"])}</div>'
        f'<p class="v-meaning">{v["meaning"]}</p>{conf_html}</section>',
        unsafe_allow_html=True)
    if p is not None:
        st.caption("Confidence is the AI's own estimate of how sure it is, not a measured accuracy rate.")

    # Duplicate-claim notice
    earlier = get_item(r.get("similar_to")) if r.get("similar_to") else None
    if earlier and ctx in ("img", "vid", "txt"):
        st.info(f"You checked a very similar claim earlier ({earlier['time']}, verdict: {earlier['assessment']}). "
                "An earlier verdict isn't proof that it still holds.", icon=":material/history:")
        c1, c2 = st.columns(2)
        c1.button("Open earlier result", key=f"{ctx}_earlier", on_click=open_result, args=(earlier["id"],))
        c2.button("Re-run without cached answer", key=f"{ctx}_fresh", on_click=queue, args=(ctx, True),
                  disabled=st.session_state.job is not None)

    # B. Claim
    st.markdown('<div class="section-title">Claim identified</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="claim">{e(r["claim"])}</div>', unsafe_allow_html=True)
    if r.get("input_text"):
        label = {"Screenshot": "Text extracted by OCR (before AI interpretation)",
                 "Video": "Speech and on-screen text extracted from the video"}.get(
                     r["input_type"], "Message you submitted")
        with st.expander(label):
            st.text(r["input_text"])
    elif r["input_type"] in ("Screenshot", "Video"):
        st.caption(f"No extracted text was used. The AI read the claim directly from the "
                   f"{r['input_type'].lower()}.")

    # C. Explanation
    if r["reasoning"]:
        st.markdown('<div class="section-title">Why this verdict</div>', unsafe_allow_html=True)
        st.write(r["reasoning"])
    if r["advice"]:
        st.info(r["advice"], icon=":material/lightbulb:")

    # D. Evidence
    groups = [("What supports the claim", r["evidence_for"]),
              ("What contradicts the claim", r["evidence_against"])]
    has_evidence = any(items for _, items in groups) or r["missing_context"] or r["red_flags"]
    st.markdown('<div class="section-title">Evidence analysis</div>', unsafe_allow_html=True)
    if has_evidence:
        st.markdown('<p class="section-note">Summarised by the AI from its search results. '
                    'Check the sources below to confirm.</p>', unsafe_allow_html=True)
        blocks = []
        for title, items in groups:
            if items:
                lis = "".join(f"<li>{e(x)}</li>" for x in items)
                blocks.append(f'<div class="ev-group"><h4>{title}</h4><ul>{lis}</ul></div>')
        if r["missing_context"]:
            blocks.append(f'<div class="ev-group"><h4>Missing context</h4><p>{e(r["missing_context"])}</p></div>')
        if r["red_flags"]:
            chips = "".join(f'<span class="flag">{e(f)}</span>' for f in r["red_flags"])
            blocks.append(f'<div class="ev-group"><h4>Warning signs in the original</h4>{chips}</div>')
        st.markdown("".join(blocks), unsafe_allow_html=True)
    else:
        st.caption("The AI didn't return a breakdown of evidence for this check.")
    if r["assessment"] == "Suspicious" and not r["evidence_for"] and not r["evidence_against"]:
        st.warning("The claim could not be sufficiently verified. Missing evidence is not proof that it is false.",
                   icon=":material/info:")

    # E. Sources
    st.markdown(f'<div class="section-title">Sources checked ({len(r["sources"])})</div>', unsafe_allow_html=True)
    if r["sources"]:
        cards = []
        for i, s in enumerate(r["sources"], 1):
            cards.append(f'<div class="src-card"><div class="n">Source {i}</div>'
                         f'<div class="pub">{e(s.get("title") or "Untitled source")}</div>'
                         f'<a href="{e(s["url"], quote=True)}" target="_blank" rel="noopener noreferrer">'
                         f'Open original page<span class="sr"> (opens in a new tab)</span></a></div>')
        st.markdown(f'<div class="src-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
        if r.get("engine", "").startswith("Offline"):
            st.caption("These are the news headlines and Wikipedia pages the offline verifier used as evidence.")
        elif r.get("source_kind") == "news":
            st.caption("These are recent news headlines Radian AI found with its own search and gave to the AI "
                       "as evidence. Open them to check what they say.")
        else:
            st.caption("These are the pages the AI retrieved with Google Search. Links pass through Google's "
                       "redirect to the original site. Whether each source supports or contradicts the claim "
                       "is described in the evidence analysis, not per source.")
    else:
        reason = ("The offline verifier found no matching news or Wikipedia sources."
                  if r.get("engine", "").startswith("Offline") else
                  "Live web search is switched off in this version, so the verdict is based on the AI model's "
                  "own knowledge." if not USE_GOOGLE_SEARCH else "No web sources were returned for this check.")
        st.info(reason + " Treat it with extra caution, especially for very recent news.",
                icon=":material/link_off:")

    # F. Metadata
    st.markdown(
        f'<dl class="meta"><div><dt>Checked</dt><dd>{e(r["time"])}</dd></div>'
        f'<div><dt>Input</dt><dd>{e(r["input_type"])}</dd></div>'
        f'<div><dt>Checked by</dt><dd>{e(r.get("engine", "Gemini"))}</dd></div>'
        f'<div><dt>Verification ID</dt><dd>{e(r["id"])}</dd></div></dl>',
        unsafe_allow_html=True)

    # G. Next actions
    st.markdown('<div class="section-title">Share or save</div>', unsafe_allow_html=True)
    st.caption("Use the copy icon on a box to copy its text.")
    st.markdown("**Verdict summary**")
    st.code(make_summary(r), language=None, wrap_lines=True)
    if r["reply_message"]:
        st.markdown("**Reply you can send to the group**")
        st.code(r["reply_message"], language=None, wrap_lines=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download report", make_report(r), file_name=f"radian-{r['id']}.txt",
                       icon=":material/download:", key=f"{ctx}_dl_{r['id']}")
    up_label = "👍 Correct" + (" (saved)" if r["feedback"] == "correct" else "")
    down_label = "👎 Wrong" + (" (saved)" if r["feedback"] == "wrong" else "")
    c2.button(up_label, key=f"{ctx}_up_{r['id']}", on_click=set_feedback, args=(r["id"], "correct"))
    c3.button(down_label, key=f"{ctx}_down_{r['id']}", on_click=set_feedback, args=(r["id"], "wrong"))

    if ctx == "img":
        st.button("Check another screenshot", key="another_img", icon=":material/add_photo_alternate:",
                  on_click=remove_image, type="secondary")
    elif ctx == "vid":
        st.button("Check another video", key="another_vid", icon=":material/video_library:",
                  on_click=remove_video, type="secondary")
    elif ctx == "txt":
        st.button("Check another message", key="another_txt", icon=":material/edit_note:",
                  on_click=clear_message, type="secondary")

    if st.session_state.get("tech"):
        with st.expander("Technical details"):
            st.write(f"Checked by: `{r.get('engine', MODEL)}`")
            st.json({k: r.get(k) for k in ("id", "assessment", "confidence", "claim", "evidence_for",
                                           "evidence_against", "missing_context", "red_flags", "sources")})


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown('<div class="brand-name">Radian AI</div><div class="brand-tag">Verify before you amplify.</div>',
                unsafe_allow_html=True)

    NAV = [("verify", "New verification", ":material/fact_check:"),
           ("history", "Verification history", ":material/history:"),
           ("about", "About Radian AI", ":material/info:")]
    for key, label, icon in NAV:
        st.button(label, key=f"nav_{key}", icon=icon, width="stretch",
                  type="primary" if st.session_state.page == key else "tertiary",
                  on_click=go, args=(key,))

    with st.expander("Advanced options", icon=":material/tune:"):
        st.toggle("Read text with OCR (Tesseract)", value=True, key="use_ocr",
                  help="Turn off to let the AI read screenshots directly without OCR.")
        st.toggle("Offline mode (skip Gemini)", value=False, key="offline_only", disabled=not OFFLINE_AVAILABLE,
                  help="Check claims with Radian AI's own offline verifier instead of Gemini."
                       if OFFLINE_AVAILABLE else
                       "The offline verifier runs on the full local install. This online version uses Gemini only.")
        st.toggle("Show technical details", value=False, key="tech",
                  help="Shows the model name, raw result fields and error details.")

    hist = st.session_state.history
    st.markdown('<p class="side-label">This session</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="stats"><div><b>{len(hist)}</b><span>Checks</span></div>'
        f'<div><b>{sum(1 for h in hist if h["assessment"] == "Likely Fake")}</b><span>Likely fake</span></div></div>',
        unsafe_allow_html=True)
    if hist:
        st.button("Clear history", key="clear_hist", icon=":material/delete:", type="tertiary",
                  on_click=clear_history)

    with st.expander("Project team", icon=":material/group:"):
        for name, role in TEAM:
            st.markdown(f"**{name}**  \n{role}")

    st.caption("Radian AI is an AI-assisted verification aid, not a final verdict. Always check the sources.")


# =====================================================================
# Header
# =====================================================================
st.markdown(
    '<header><div class="app-header"><h1 class="app-title">Radian AI</h1>'
    '<span class="ai-badge">AI-ASSISTED VERIFICATION</span></div>'
    '<p class="app-sub">Verify forwarded messages, screenshots and videos with evidence-based AI assistance.</p>'
    '</header>', unsafe_allow_html=True)

busy = st.session_state.job is not None


# =====================================================================
# Page: New verification
# =====================================================================
def page_verify():
    tab_img, tab_vid, tab_txt = st.tabs([":material/image: Check a screenshot", ":material/videocam: Check a video",
                                         ":material/chat: Check a message"])

    # ---------------- Screenshot ----------------
    with tab_img:
        img = st.session_state.img
        r = get_item(st.session_state.img_result_id)
        if r:
            extract = "done" if r.get("used_ocr") else "skipped"
        else:
            extract = "current" if img else "todo"
        later = "done" if r else "todo"
        step_bar([("Upload", "done" if img else "current"), ("Extract text", extract),
                  ("Analyze claim", later), ("Check sources", later), ("Verdict", "current" if r else "todo")])

        left, right = st.columns([1, 1.45], gap="large")
        with left:
            st.markdown('<div class="panel-title">Upload a screenshot</div>'
                        '<p class="panel-help">Drag and drop an image here, or browse files. '
                        f'PNG, JPG or WEBP, up to {st.get_option("server.maxUploadSize")} MB.</p>',
                        unsafe_allow_html=True)
            f = st.file_uploader("Upload a screenshot", type=IMAGE_TYPES, key=f"uploader_{st.session_state.up_key}",
                                 label_visibility="collapsed", disabled=busy,
                                 help="WhatsApp forwards, tweets, news headlines or Instagram posts")
            if f is not None:
                fid = f"{f.name}-{f.size}"
                if st.session_state.up_seen != fid:
                    st.session_state.up_seen = fid
                    data = f.getvalue()
                    try:
                        Image.open(io.BytesIO(data)).verify()
                        st.session_state.img = {"bytes": data, "name": f.name, "type": f.type or "image/png",
                                                "size": f.size, "fid": fid}
                        st.session_state.img_invalid = None
                    except Exception:
                        st.session_state.img = None
                        st.session_state.img_invalid = f.name
                    st.session_state.img_result_id = None
                    st.session_state.errors.pop("img", None)
                    st.rerun()
            elif st.session_state.up_seen is not None:
                # The user removed the file with the upload widget's own remove button.
                remove_image()
                st.rerun()

            img = st.session_state.img
            if st.session_state.img_invalid:
                st.error(f"“{st.session_state.img_invalid}” couldn't be read as an image. "
                         "Upload a PNG, JPG or WEBP screenshot.", icon=":material/broken_image:")
            if img:
                b64 = base64.b64encode(img["bytes"]).decode()
                st.markdown(f'<div class="preview"><img src="data:{img["type"]};base64,{b64}" '
                            f'alt="Preview of the uploaded screenshot"><p>{html.escape(img["name"])} '
                            f'({img["size"] / 1024:,.0f} KB)</p></div>', unsafe_allow_html=True)
                st.button("Remove image", key="remove_img", icon=":material/close:", type="tertiary",
                          on_click=remove_image, disabled=busy)

            st.button("Analyze screenshot", key="btn_img", type="primary", icon=":material/search:",
                      width="stretch", disabled=busy or not img, on_click=queue, args=("img",))
            if not img and not st.session_state.img_invalid:
                st.caption("Upload a screenshot to enable analysis.")

        with right:
            process_job("img")
            r = get_item(st.session_state.img_result_id)
            if show_error("img"):
                pass
            elif r:
                render_result(r, "img")
            elif img:
                empty_state("Ready to analyze", "Click <i>Analyze screenshot</i>. The verdict, evidence and sources "
                            "will appear here.")
            else:
                empty_state("No screenshot yet", "Upload a screenshot to get started.")

    # ---------------- Video ----------------
    with tab_vid:
        vid = st.session_state.vid
        r = get_item(st.session_state.vid_result_id)
        if r:
            extract = "done" if r.get("used_ocr") else "skipped"
        else:
            extract = "current" if vid else "todo"
        later = "done" if r else "todo"
        step_bar([("Upload", "done" if vid else "current"), ("Extract speech & text", extract),
                  ("Analyze claim", later), ("Check sources", later), ("Verdict", "current" if r else "todo")])

        left, right = st.columns([1, 1.45], gap="large")
        with left:
            limit = (f"MP4, MOV or WEBM, up to {VIDEO_MAX_MB} MB (about a minute). " if VIDEO_MAX_MB
                     else "A news clip, reel or forwarded video, up to 5 minutes. ")
            st.markdown('<div class="panel-title">Upload a video</div>'
                        f'<p class="panel-help">{limit}'
                        'Radian AI reads what is said and shown, then checks those claims. '
                        'It does not detect deepfakes.</p>', unsafe_allow_html=True)
            f = st.file_uploader("Upload a video", type=VIDEO_TYPES, key=f"vid_uploader_{st.session_state.vid_up_key}",
                                 label_visibility="collapsed", disabled=busy,
                                 help=", ".join(t.upper() for t in VIDEO_TYPES))
            if f is not None:
                fid = f"{f.name}-{f.size}"
                if st.session_state.vid_seen != fid:
                    st.session_state.vid_seen = fid
                    st.session_state.vid = {"bytes": f.getvalue(), "name": f.name,
                                            "type": f.type or "video/mp4", "size": f.size, "fid": fid}
                    st.session_state.vid_result_id = None
                    st.session_state.errors.pop("vid", None)
                    st.rerun()
            elif st.session_state.vid_seen is not None:
                remove_video()
                st.rerun()

            vid = st.session_state.vid
            too_big = bool(vid and VIDEO_MAX_MB and vid["size"] > VIDEO_MAX_MB * 1024 * 1024)
            if too_big:
                st.error(f"This video is {vid['size'] / (1024 * 1024):,.1f} MB. The online version can check "
                         f"videos up to {VIDEO_MAX_MB} MB. Trim it or upload a shorter clip.",
                         icon=":material/video_file:")
            if vid:
                st.video(vid["bytes"])
                st.caption(f"{html.escape(vid['name'])} ({vid['size'] / (1024 * 1024):,.1f} MB)")
                st.button("Remove video", key="remove_vid", icon=":material/close:", type="tertiary",
                          on_click=remove_video, disabled=busy)

            st.button("Analyze video", key="btn_vid", type="primary", icon=":material/search:",
                      width="stretch", disabled=busy or not vid or too_big, on_click=queue, args=("vid",))
            if not vid:
                st.caption("Upload a video to enable analysis.")

        with right:
            process_job("vid")
            r = get_item(st.session_state.vid_result_id)
            if show_error("vid"):
                pass
            elif r:
                render_result(r, "vid")
            elif vid:
                empty_state("Ready to analyze", "Click <i>Analyze video</i>. Reading the speech and on-screen "
                            "text can take a minute or two.")
            else:
                empty_state("No video yet", "Upload a video to get started.")

    # ---------------- Message ----------------
    with tab_txt:
        if "msg_input" not in st.session_state:
            st.session_state.msg_input = st.session_state.msg_saved   # restore after switching pages
        r = get_item(st.session_state.txt_result_id)
        has_msg = len(st.session_state.msg_saved.strip()) >= 10
        later = "done" if r else "todo"
        step_bar([("Paste message", "done" if (has_msg or r) else "current"), ("Extract text", "skipped"),
                  ("Analyze claim", later if r else ("current" if has_msg else "todo")),
                  ("Check sources", later), ("Verdict", "current" if r else "todo")])

        left, right = st.columns([1, 1.45], gap="large")
        with left:
            st.markdown('<div class="panel-title">Paste a message</div>'
                        '<p class="panel-help">Paste a forwarded message, headline or post text.</p>',
                        unsafe_allow_html=True)
            st.text_area("Message to check", key="msg_input", height=220, on_change=save_message,
                         label_visibility="collapsed", disabled=busy,
                         placeholder="Example: Government is giving a free ₹749 recharge to all Indians. "
                                     "Click this link to claim before midnight!")
            count = len(st.session_state.get("msg_input", ""))
            st.caption(f"{count:,} characters")
            if st.session_state.msg_invalid:
                st.warning("Paste a longer message to check (at least 10 characters).", icon=":material/edit:")
            c1, c2 = st.columns([2, 1])
            c1.button("Analyze message", key="btn_txt", type="primary", icon=":material/search:",
                      width="stretch", disabled=busy, on_click=queue_message)
            c2.button("Clear", key="clear_txt", icon=":material/close:", width="stretch",
                      disabled=busy, on_click=clear_message)

        with right:
            process_job("txt")
            r = get_item(st.session_state.txt_result_id)
            if show_error("txt"):
                pass
            elif r:
                render_result(r, "txt")
            else:
                empty_state("No message checked yet", "Paste a message, then click "
                            "<i>Analyze message</i>.")


# =====================================================================
# Page: History
# =====================================================================
def page_history():
    st.markdown('<div class="panel-title">Verification history</div>'
                '<p class="panel-help">Stored for this browser session only. Refreshing or closing the page '
                'clears it.</p>', unsafe_allow_html=True)
    hist = st.session_state.history

    item = get_item(st.session_state.open_id)
    if item:
        st.button("Back to history", key="back_hist", icon=":material/arrow_back:", type="tertiary",
                  on_click=lambda: st.session_state.update(open_id=None))
        render_result(item, "hist")
        return

    if not hist:
        empty_state("No checks yet", "Results from this session will be listed here.")
        st.button("Start a verification", key="start_from_hist", type="primary", on_click=go, args=("verify",))
        return

    counts = {k: sum(1 for h in hist if h["assessment"] == k) for k in VERDICT}
    c1, c2 = st.columns([1, 1.4])
    query = c1.text_input("Search checks", placeholder="Search claims and messages",
                          key="hist_query", label_visibility="collapsed")
    options = ["All"] + list(VERDICT)
    choice = c2.segmented_control("Filter by verdict", options, default="All", key="hist_filter",
                                  label_visibility="collapsed",
                                  format_func=lambda o: f"All ({len(hist)})" if o == "All" else f"{o} ({counts[o]})")
    choice = choice or "All"

    q = normalize(query)
    rows = [h for h in hist
            if (choice == "All" or h["assessment"] == choice)
            and (not q or q in normalize(h["claim"]) or q in normalize(h.get("input_text")))]

    if not rows:
        empty_state("No matching checks", "Try a different search or verdict filter.")
        return

    for h in rows:
        v = VERDICT[h["assessment"]]
        with st.container(border=True):
            a, b = st.columns([4, 1], vertical_alignment="center")
            with a:
                st.badge(h["assessment"], icon=v["icon"], color=v["badge"])
                st.markdown(f"**{html.escape(h['claim'])}**")
                fb = f" · marked {h['feedback']}" if h["feedback"] else ""
                st.caption(f"{h['time']} · {h['input_type']} · {h['id']}{fb}")
            b.button("Open", key=f"open_{h['id']}", icon=":material/open_in_full:", width="stretch",
                     on_click=open_result, args=(h["id"],))


# =====================================================================
# Page: About
# =====================================================================
def page_about():
    st.markdown('<div class="panel-title">About Radian AI</div>', unsafe_allow_html=True)
    st.write("Radian AI helps people check forwarded messages, screenshots and videos before they share them. "
             "It reads the text, identifies the main claim, searches reliable sources and explains its verdict, "
             "so you can see the reasoning instead of trusting an unexplained answer.")

    st.markdown('<div class="section-title">How a check works</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Upload** a screenshot or video, or paste a message.\n"
        "2. **Extract text** from screenshots with Tesseract OCR (English and Hindi), and from videos "
        "with Whisper speech-to-text plus OCR on frames.\n"
        "3. **Analyze the claim** with Google Gemini.\n"
        "4. **Check sources** using Google Search, with priority for fact-checkers such as PIB Fact Check, "
        "Alt News and BOOM (when live search is enabled).\n"
        "5. **Verdict** with reasons, evidence, warning signs and the sources retrieved.\n\n"
        "If Gemini is unavailable (for example, the free-tier limit is reached), Radian AI switches to its "
        "**offline verifier**: it rewrites the message as a clear claim, finds evidence in recent news "
        "and Wikipedia, checks each source with a Natural Language Inference model, and adds a signal "
        "from our own trained writing-style model.")

    st.markdown('<div class="section-title">What the verdicts mean</div>', unsafe_allow_html=True)
    for label, v in VERDICT.items():
        st.markdown(f"**{v['symbol']} {label}**: {v['meaning']}")

    st.markdown('<div class="section-title">Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        "- The AI can be wrong, especially about very recent events or local news with little coverage.\n"
        "- OCR can misread low-quality, stylised or dark-mode screenshots.\n"
        "- For videos, Radian AI checks the claims that are spoken or shown on screen. It does not "
        "detect deepfakes or edited footage.\n"
        "- Confidence is the AI's own estimate, not a measured accuracy rate.\n"
        "- History is kept only for the current browser session.")

    st.markdown('<div class="section-title">Project team</div>', unsafe_allow_html=True)
    st.markdown("\n".join(f"- **{n}**: {r}" for n, r in TEAM))
    st.caption("Built with Streamlit, the Google Gemini API, Tesseract OCR, faster-whisper, Hugging Face "
               "Transformers and scikit-learn.")


{"verify": page_verify, "history": page_history, "about": page_about}[st.session_state.page]()
