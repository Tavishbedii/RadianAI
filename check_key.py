import re
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

client = genai.Client()
search = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])

for n in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-flash-latest"]:
    for label, cfg in [("WITHOUT search", None), ("WITH search", search)]:
        try:
            client.models.generate_content(model=n, contents="Reply with OK", config=cfg)
            print("WORKS ->", n, label)
        except Exception as e:
            msg = str(e)
            found = re.findall(r"'quotaId': '([^']+)'.*?'quotaValue': '(\d+)'", msg)
            print("fails ->", n, label, "|", found if found else msg[:250])