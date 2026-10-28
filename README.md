# 🔎 Radian AI

**Verify before you amplify.**

Radian AI is an AI-powered fake news detection app for the kind of news people in India receive every day on WhatsApp and social media. Paste a forwarded message, upload a screenshot, or upload a short video, and Radian AI extracts the claim, checks it against evidence, and explains its verdict with sources.

It uses **Google Gemini** as its main engine and automatically switches to **its own offline verification system** when Gemini is unavailable, so a check never just fails because of an API limit.

> ⚠️ Radian AI is an AI-assisted verification aid, not a final verdict. Always check the sources yourself.

---

## ✨ Features

- **Check a message**: paste a forwarded message, headline, or post. Typos, questions, and personal stories are handled ("is diwali on 8 nov??" is checked as "Diwali is on 8 November").
- **Check a screenshot**: upload a PNG, JPG, or WEBP image. Text is read with Tesseract OCR (English and Hindi), and social-media clutter such as likes, timestamps, and usernames is removed.
- **Check a video**: upload a clip of up to 5 minutes. Speech is transcribed with Whisper (Hindi and other languages are translated to English) and on-screen text is read from frames.
- **Clear verdicts**: every check ends as **Credible**, **Suspicious**, or **Likely Fake**, with reasoning, supporting and contradicting evidence, warning signs, and source links.
- **Automatic offline fallback**: if Gemini fails (for example, the free-tier limit is reached), Radian AI uses its own claim cleaner, evidence verifier, and trained model instead.
- **Offline mode**: a sidebar switch to run checks with Radian AI's own system only.
- **Transparency**: each result shows the exact claim checked and which engine produced the verdict ("Checked by").
- **Ready-to-send reply**: a short, polite message the user can send back to the group.
- **Session history and feedback**: review earlier checks and mark verdicts as correct or wrong.

---

## ⚙️ How It Works

```
Message / Screenshot / Video
        │
        ▼
Extract text
  • Screenshot → Tesseract OCR (English + Hindi)
  • Video      → faster-whisper (speech → English) + OCR on frames every 3 seconds
        │
        ▼
Gemini available? ──── YES ──► Gemini analyses the claim (optionally with Google Search)
        │
        NO (quota reached, no key, or Offline mode on)
        │
        ▼
Radian AI offline verifier
  1. Claim cleaner     → turns the message into a clear, checkable claim
  2. Evidence search   → recent Google News headlines + Wikipedia sentences
  3. NLI model         → does each source support or contradict the claim?
  4. Style model       → does the writing resemble known fake news?
  5. Warning signs     → "share before it's deleted", ALL CAPS, short links…
        │
        ▼
Verdict: Credible / Suspicious / Likely Fake  +  evidence  +  sources
```

The offline verifier only gives a firm verdict when **at least two sources agree**, ignores question headlines ("Is Diwali on November 9?"), and answers **Suspicious** when the evidence is too weak, rather than guessing.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Interface | Streamlit, custom HTML/CSS |
| Primary AI | Google Gemini (`gemini-3.5-flash-lite`) via the `google-genai` SDK, optional Google Search grounding |
| Claim verification (offline) | Hugging Face Transformers + PyTorch with the DeBERTa-v3 NLI model `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Trained fake news model | scikit-learn (TF-IDF + Linear SVM), pandas, joblib |
| Evidence sources | Google News RSS (Indian English news, last 180 days), Wikipedia API |
| Screenshots | Tesseract OCR 5 (English + Hindi) via `pytesseract`, Pillow |
| Videos | faster-whisper (speech-to-text and translation), OpenCV (frame extraction) |
| Configuration | python-dotenv (`.env` file) |
| Language and tools | Python 3.12, venv, VS Code, Git and GitHub |

---

## 🧠 Our Trained Model

Radian AI includes a fake news classifier we trained ourselves on the Kaggle **Fake and Real News Dataset** (by Clément Bisaillon).

| Model | Test accuracy |
|---|---|
| **Linear SVM** (selected) | **98.65%** |
| Logistic Regression | 97.63% |
| Naive Bayes | 94.72% |

- 38,822 articles after removing duplicates and empty rows (17,900 fake, 20,922 real)
- 80/20 train/test split; tested on 7,765 unseen articles
- Precision 0.99 and recall 0.98 for fake news
- The word "Reuters" is removed during training, because every real article in the dataset contains it and the model would otherwise learn a shortcut

**Important:** this model learns *writing style*, not facts, and was trained on US news articles from about 2015–2018. It is used only as an extra signal. Short messages get "Not sure" instead of a guess, and factual checking is done by the evidence verifier.

---

## 📁 Project Structure

```
RadianAI/
├── app.py                  # Streamlit app: interface, Gemini, offline fallback
├── offline_check.py        # Combines Radian AI's own components into one result
├── claim_cleaner.py        # Turns messy messages into clear, checkable claims
├── verify_claim.py         # Evidence search + NLI verification (also runs in the terminal)
├── news_search.py          # Google News RSS lookup
├── video_reader.py         # Whisper speech-to-text + OCR on video frames
├── train_model.py          # Trains the fake news classifier
├── predict.py              # Uses the trained classifier (also runs in the terminal)
├── check_key.py            # Tests your Gemini API key
├── fake_news_model.joblib  # Trained model (created by train_model.py)
├── requirements.txt        # Python dependencies
├── data/                   # Fake.csv and True.csv (not uploaded to GitHub)
├── .env                    # API key and settings (never uploaded to GitHub)
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Tavishbedii/RadianAI.git
cd RadianAI
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

On Windows, PyTorch also needs the **Microsoft Visual C++ Redistributable (x64)**: https://aka.ms/vs/17/release/vc_redist.x64.exe

### 4. Install Tesseract OCR (for screenshots and video text)

- Windows: install from the official releases (https://github.com/tesseract-ocr/tesseract/releases) into `C:\Program Files\Tesseract-OCR`.
- Add Hindi: download `hin.traineddata` from https://github.com/tesseract-ocr/tessdata and copy it into `C:\Program Files\Tesseract-OCR\tessdata`.
- Check with: `tesseract --list-langs` (you should see `eng` and `hin`).

If Tesseract is installed elsewhere, update `TESSERACT_PATH` in `app.py` and `video_reader.py`.

### 5. Add your Gemini API key

Create a free key in [Google AI Studio](https://aistudio.google.com/), then create a file named `.env` in the project folder:

```
GEMINI_API_KEY=your-api-key-here
USE_GOOGLE_SEARCH=false
OFFLINE_FALLBACK=true
```

Test the key with `python check_key.py`.

### 6. Train the fake news model (one time)

1. Download the **Fake and Real News Dataset** from Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Put `Fake.csv` and `True.csv` in a folder named `data/`.
3. Run:

```bash
python train_model.py
```

This creates `fake_news_model.joblib`.

### 7. Run the app

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501**. The first offline check downloads the NLI model (about 700 MB) and the first video check downloads the Whisper model (about 150 MB).

### Optional: terminal tools

```bash
python predict.py        # writing-style model only
python verify_claim.py   # full offline claim verifier
```

---

## 🔧 Configuration

| Setting | Where | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | `.env` | – | Your Gemini API key |
| `USE_GOOGLE_SEARCH` | `.env` | `false` | `true` lets Gemini search live news (needs a plan that includes Google Search grounding) |
| `OFFLINE_FALLBACK` | `.env` | `true` | Use Radian AI's offline verifier when Gemini fails |
| `WHISPER_MODEL` | `.env` | `base` | Speech model size: `tiny`, `base` or `small` (larger is slower but more accurate) |
| `MODEL` | `app.py` | `gemini-3.5-flash-lite` | Gemini model used for analysis |
| `TESSERACT_PATH` | `app.py`, `video_reader.py` | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Location of Tesseract |
| `GEMINI_VIDEO_LIMIT_MB` | `app.py` | `18` | Videos up to this size are sent to Gemini directly; larger ones are sent as extracted text |

Sidebar **Advanced options**: OCR on/off, **Offline mode (skip Gemini)**, and technical details.

---

## ⚠️ Limitations

- **Recent news:** without Google Search grounding, Gemini doesn't know events after its training cutoff. The offline verifier helps by searching recent news.
- **Evidence from headlines:** the offline verifier reads headlines and Wikipedia introductions, not full articles. When sources don't state a fact clearly, the verdict is **Suspicious**.
- **Hindi offline:** OCR can read Hindi, but without Gemini, Hindi text can't be translated, so Hindi screenshots are usually marked Suspicious offline. Hindi *speech* in videos works offline, because Whisper translates it.
- **Videos:** Radian AI checks the *claims* spoken or shown in a video. It does **not** detect deepfakes, edited footage, or old videos reused out of context.
- **Trained model:** learns writing style from US news, so it is only an extra signal, not a fact-checker.
- **OCR quality:** blurry, stylised, or dark-mode images may be misread.
- **Free-tier limits:** the Gemini free tier limits requests; the offline verifier takes over when the limit is reached.
- **Not a final authority:** AI can make mistakes. Radian AI shows its reasoning and sources so users can verify for themselves.

---

## 🔮 Future Scope

- Retrain the style model on Indian and Hindi news datasets
- Add trusted fact-checkers (PIB Fact Check, Alt News, BOOM) as evidence sources
- Read full articles, not just headlines, for stronger offline evidence
- Detect manipulated images, deepfakes, and old footage reused out of context
- A WhatsApp bot and browser extension for one-tap verification

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

Never commit your `.env` file or API key to GitHub. The `.gitignore` keeps `.env`, the virtual environment, and the dataset out of the repository.
