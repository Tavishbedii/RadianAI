# 🔎 Radian AI

**Verify before you amplify.**

Radian AI is an AI-powered fake news detection app. Paste a forwarded message or upload a screenshot of a post, and Radian AI extracts the text, analyzes the claim, checks it against sources, and gives a clear credibility verdict with an explanation.

> ⚠️ Radian AI is an AI-assisted verification aid, not a final verdict. Always check the sources yourself.

---

## ✨ Features

- **Check a screenshot**: upload a PNG, JPG, JPEG or WEBP image of a WhatsApp forward, tweet, or news post. The app reads the text from the image automatically.
- **Check a message**: paste a forwarded message, headline, or post text directly.
- **Step-by-step pipeline**: every check moves through five visible stages: input, text extraction, claim analysis, source checking, and verdict.
- **Live web verification (optional)**: with Google Search grounding enabled, Gemini checks claims against current news, so recent events are handled correctly.
- **Honest uncertainty**: when web search is off, the model is instructed to say a claim may be too recent to verify rather than guessing.
- **Verification history**: review earlier checks from the current session.
- **Session stats**: see how many claims were checked and how many were flagged as likely fake.

---

## ⚙️ How It Works

```
User input (screenshot or text)
        │
        ▼
01  Upload screenshot / paste message
        │
        ▼
02  Extract text from the image (skipped for pasted text)
        │
        ▼
03  Analyze the claim with Gemini
        │
        ▼
04  Check sources (Google Search grounding, if enabled)
        │
        ▼
05  Verdict + explanation + sources
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| AI model | Google Gemini (`gemini-3.5-flash-lite`) via the `google-genai` SDK |
| Text extraction | Gemini vision, with an optional Tesseract OCR path |
| Live fact-checking | Gemini Google Search grounding (optional) |
| Image handling | Pillow (PIL) |
| Configuration | python-dotenv (`.env` file) |
| Language | Python |

---

## 📁 Project Structure

```
RadianAI/
├── app.py              # Main Streamlit application
├── check_key.py        # Helper script to test your Gemini API key
├── config              # Configuration file
├── requirements.txt    # Python dependencies
├── .env                # API key and settings (NOT uploaded to GitHub)
└── .gitignore          # Keeps .env and cache files out of Git
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Tavishbedii/RadianAI.git
cd RadianAI
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Get a Gemini API key

Create a free API key in [Google AI Studio](https://aistudio.google.com/).

### 5. Create a `.env` file

In the project folder, create a file named `.env`:

```
GEMINI_API_KEY=your-api-key-here
USE_GOOGLE_SEARCH=false
```

Set `USE_GOOGLE_SEARCH=true` only if your Gemini plan includes Google Search grounding.

You can test your key with:

```bash
python check_key.py
```

### 6. Run the app

```bash
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🔧 Configuration

| Setting | Where | Description |
|---|---|---|
| `GEMINI_API_KEY` | `.env` | Your Gemini API key |
| `USE_GOOGLE_SEARCH` | `.env` | `true` enables live Google Search grounding; default is `false` |
| `MODEL` | `app.py` | The Gemini model used for analysis |
| `TESSERACT_PATH` | `app.py` | Path to Tesseract OCR, if installed |
| `IMAGE_TYPES` | `app.py` | Allowed upload formats (`png`, `jpg`, `jpeg`, `webp`) |

---

## ⚠️ Limitations

- **Recent events**: without Google Search grounding, the model cannot know about news after its training cutoff. Enable search for up-to-date checks.
- **Free-tier limits**: the Gemini free tier has request limits. If you see "free-tier limit reached", wait a minute and try again.
- **Text extraction**: blurry, low-resolution, or heavily styled images may be read incorrectly.
- **Not a final authority**: AI can make mistakes. Radian AI shows its reasoning and sources so users can verify claims themselves.

---

## 🔮 Future Scope

- Support for more Indian languages
- Detection of manipulated or AI-generated images
- A browser extension and WhatsApp bot for one-tap verification
- Integration with trusted fact-checking databases

---

## 👥 Team

| Name | Role |
|---|---|
| Shubham | Backend & bot |
| Siddharth | OCR |
| Tavish Bedi | AI model |
| Swayam | Frontend |

---

## 🔒 Security Note

Never commit your `.env` file or API key to GitHub. The included `.gitignore` keeps `.env` out of the repository.
