"""Free latest-news lookup for Radian AI.

Uses the Google News RSS feed (no API key, no Gemini quota) to find recent
headlines about a claim, so Gemini can judge recent events it was never trained on.
"""
import re
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

STOPWORDS = {
    "i", "me", "my", "we", "you", "he", "she", "it", "they", "is", "am", "are",
    "was", "were", "be", "been", "the", "a", "an", "and", "or", "but", "of", "to",
    "in", "on", "at", "for", "with", "this", "that", "these", "those", "saw",
    "seen", "yesterday", "today", "just", "really", "very", "please", "share",
    "forward", "forwarded", "news", "breaking", "viral", "true", "fake",
}


def _keywords(claim, limit):
    words = re.findall(r"\w+", claim.lower())
    keep = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return " ".join(keep[:limit])


def _search(query, max_results):
    url = (f"https://news.google.com/rss/search?q={quote(query)}"
           "&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return []
    return [
        {
            "title": item.findtext("title", ""),
            "link": item.findtext("link", ""),
            "date": item.findtext("pubDate", ""),
        }
        for item in root.findall("./channel/item")[:max_results]
    ]


def fetch_news(claim, max_results=5):
    """Search with up to 6 keywords; if nothing is found, retry with 3."""
    for limit in (6, 3):
        query = _keywords(claim, limit)
        if not query:
            return []
        results = _search(query, max_results)
        if results:
            return results
    return []


def news_context(claim):
    """Text block to append to the Gemini prompt."""
    articles = fetch_news(claim)
    if not articles:
        return "\n\nNo recent news articles were found for this claim."
    lines = [f"- {a['title']} ({a['date']}) {a['link']}" for a in articles]
    return (
        "\n\nRecent news headlines found for this claim. Treat these as more "
        "up to date than your own knowledge, use them as evidence, and cite "
        "them in your answer where relevant:\n" + "\n".join(lines)
    )