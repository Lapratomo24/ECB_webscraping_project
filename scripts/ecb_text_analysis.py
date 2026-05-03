from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, WordCloud

# Step 1: Store the ECB webpage address.
URL = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2026/html/ecb.is260319~93b1cbad97.en.html"

# Step 2: Define folders for saved text and output files.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Step 3: Create the folders if they do not exist yet.
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def clean_whitespace(text: str) -> str:
    """Turn repeated spaces, tabs, and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()

def sentiment_label(compound: float) -> str:
    """Convert a VADER compound score into a simple label."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"

def tokenize_words(text: str, stopwords: set[str]) -> list[str]:
    """Make a simple list of lowercase words, excluding stopwords."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    return [token for token in tokens if len(token) > 2 and token not in stopwords]

# Step 4: Define a new helper function to analyze and save data for any section of text.
def analyze_text_section(text: str, section_name: str, stopwords: set[str], analyzer: SentimentIntensityAnalyzer):
    """Runs sentiment analysis, counts words, generates a word cloud, and saves all outputs."""
    print(f"--- Processing {section_name.upper()} ---")

    # 4a. Save the raw text
    text_path = DATA_DIR / f"ecb_{section_name}_2026-03-19.txt"
    text_path.write_text(text, encoding="utf-8")

    # 4b. Sentiment Analysis
    scores = analyzer.polarity_scores(text)
    scores["sentiment_label"] = sentiment_label(scores["compound"])
    scores["section"] = section_name
    scores["source_url"] = URL

    sentiment_summary = pd.DataFrame([scores])
    sentiment_path = OUTPUT_DIR / f"ecb_{section_name}_sentiment.csv"
    sentiment_summary.to_csv(sentiment_path, index=False)

    # 4c. Tokenize and Count Words
    tokens = tokenize_words(text, stopwords)
    word_counts = Counter(tokens)

    top_words = pd.DataFrame(word_counts.most_common(30), columns=["word", "count"])
    top_words_path = OUTPUT_DIR / f"ecb_{section_name}_top_words.csv"
    top_words.to_csv(top_words_path, index=False)

    # 4d. Generate and Save Word Cloud
    wordcloud = WordCloud(
        width=1200, height=700, background_color="white",
        stopwords=stopwords, colormap="viridis", random_state=42
    ).generate(text)

    wordcloud_path = OUTPUT_DIR / f"ecb_{section_name}_wordcloud.png"
    plt.figure(figsize=(12, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(wordcloud_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Print brief summary to the console
    print(f"Sentiment: {scores['sentiment_label']} (Compound: {scores['compound']})")
    print(f"Saved files: Text, Sentiment CSV, Top Words CSV, Wordcloud PNG to folders.\n")


# Step 5: Setup requests and download the page.
headers = {"User-Agent": "Mozilla/5.0 (beginner text analysis tutorial)"}
response = requests.get(URL, headers=headers, timeout=30)
response.raise_for_status()

# Step 6: Parse the HTML.
soup = BeautifulSoup(response.text, "lxml")
section = soup.select_one("main div.section")
if section is None:
    raise RuntimeError("Could not find the article section.")

# Step 7: Clean out unwanted elements.
# Note: We keep `a[href="#qa"]` in the decompose list because that's the "jump link" at the top, not the actual section anchor.
for unwanted in section.select('script, style, a[href="#qa"], .ecb-publicationDate'):
    unwanted.decompose()

# Step 8: Prepare lists to hold our two separate blocks of text, and a flag to track where we are.
statement_blocks = []
qa_blocks = []
in_qa_section = False

# Step 9: Iterate through headings and paragraphs.
for element in section.find_all(["h2", "p"]):
    classes = element.get("class", [])

    if "ecb-pressContentSubtitle" in classes:
        continue

    # Step 10: Check if we have reached the Q&A anchor.
    # The anchor target might be an id="qa" on the heading, or an <a name="qa"> inside it.
    if element.get("id") == "qa" or element.find(id="qa") or element.find(attrs={"name": "qa"}):
        in_qa_section = True  # Flip the switch! Everything after this goes to QA.

    text = clean_whitespace(element.get_text(" ", strip=True))
    if text:
        # Step 11: Route the text to the correct list based on our flag.
        if in_qa_section:
            qa_blocks.append(text)
        else:
            statement_blocks.append(text)

# Step 12: Join the blocks into two full documents.
statement_text = "\n\n".join(statement_blocks)
qa_text = "\n\n".join(qa_blocks)

# Step 13: Initialize our NLP tools (VADER and Stopwords).
nltk.download("vader_lexicon", quiet=True)
vader_analyzer = SentimentIntensityAnalyzer()

custom_stopwords = set(STOPWORDS)
custom_stopwords.update({
    "ecb", "euro", "area", "monetary", "policy", "inflation", "per", "cent",
    "will", "would", "could", "also", "question", "questions", "answer",
    "answers", "think", "going",
})

# Step 14: Run our new helper function on both sections!
if statement_text:
    analyze_text_section(statement_text, "statement", custom_stopwords, vader_analyzer)
else:
    print("Warning: No text found for the Statement.")

if qa_text:
    analyze_text_section(qa_text, "qa", custom_stopwords, vader_analyzer)
else:
    print("Warning: No text found for the Q&A section.")

print("Analysis complete!")
