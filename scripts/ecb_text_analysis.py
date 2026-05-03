from pathlib import Path
import re

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer


# Step 1: Store the ECB webpage address.
URL = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2026/html/ecb.is260319~93b1cbad97.en.html"

# Step 2: Define folders for saved data and outputs.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Step 3: Create the folders if they are missing.
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_whitespace(text: str) -> str:
    # Step 4: Replace repeated spaces and line breaks with a single space.
    """Turn repeated spaces, tabs, and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def sentiment_label(compound: float) -> str:
    # Step 5: Convert the numeric VADER compound score into a simple label.
    # These cutoffs are the common beginner-friendly VADER rules.
    """Convert a VADER compound score into a simple label."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


# Step 6: Add a User-Agent header so the request looks browser-like.
headers = {
    "User-Agent": "Mozilla/5.0 (beginner text analysis tutorial)"
}

# Step 7: Download the page.
response = requests.get(URL, headers=headers, timeout=30)

# Step 8: Stop if the download failed.
response.raise_for_status()

# Step 9: Parse the downloaded HTML.
soup = BeautifulSoup(response.text, "lxml")

# Step 10: Find the main container that holds the press conference content.
section = soup.select_one("main div.section")
if section is None:
    raise RuntimeError("Could not find the article section. The page structure may have changed.")

# Step 11: Remove pieces of the page that we do not want in the text analysis.
for unwanted in section.select('script, style, a[href="#qa"], .ecb-publicationDate'):
    unwanted.decompose()

# Step 12: Collect clean text blocks from headings and paragraphs.
text_blocks = []
for element in section.find_all(["h2", "p"]):
    classes = element.get("class", [])

    # Skip the subtitle line with names and roles.
    if "ecb-pressContentSubtitle" in classes:
        continue

    # Convert the HTML element into plain text.
    text = clean_whitespace(element.get_text(" ", strip=True))
    if text:
        text_blocks.append(text)

# Step 13: Combine all blocks into one document.
full_text = "\n\n".join(text_blocks)

# Step 14: Save the clean full script so we have the source text on disk.
text_path = DATA_DIR / "ecb_press_conference_2026-03-19.txt"
text_path.write_text(full_text, encoding="utf-8")

# Step 15: Download the VADER lexicon.
# This is the word list NLTK uses behind the scenes for sentiment scoring.
nltk.download("vader_lexicon", quiet=True)

# Step 16: Create the sentiment analyzer object.
sentiment_analyzer = SentimentIntensityAnalyzer()

# Step 17: Score the entire press conference as one text.
# The result is a dictionary with neg, neu, pos, and compound values.
scores = sentiment_analyzer.polarity_scores(full_text)

# Step 18: Add a simple label and keep the source URL with the result.
scores["sentiment_label"] = sentiment_label(scores["compound"])
scores["source_url"] = URL

# Step 19: Turn the dictionary into a one-row table.
# We use pandas so it is easy to save and inspect the result.
sentiment_summary = pd.DataFrame([scores])

# Step 20: Save the summary table as a CSV file.
sentiment_path = OUTPUT_DIR / "ecb_sentiment_summary.csv"
sentiment_summary.to_csv(sentiment_path, index=False)

# Step 21: Print a short report for the user.
print("Saved text to:", text_path)
print("Saved sentiment summary to:", sentiment_path)
print()
print("Whole-document sentiment scores:")
print(sentiment_summary[["neg", "neu", "pos", "compound", "sentiment_label"]])
