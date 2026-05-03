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
    # Step 4: Turn messy spacing into clean spacing.
    """Turn repeated spaces, tabs, and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def sentiment_label(compound: float) -> str:
    # Step 5: Map the numeric compound score to a simple word label.
    """Convert a VADER compound score into a simple label."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def tokenize_words(text: str, stopwords: set[str]) -> list[str]:
    # Step 6: Break the text into lowercase word tokens.
    # We keep alphabetic words and allow apostrophes or hyphens inside them.
    # Then we remove short words and stopwords.
    """Make a simple list of lowercase words, excluding stopwords."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    return [
        token
        for token in tokens
        if len(token) > 2 and token not in stopwords
    ]


# Step 7: Add a User-Agent so the request looks browser-like.
headers = {
    "User-Agent": "Mozilla/5.0 (beginner text analysis tutorial)"
}

# Step 8: Download the ECB page.
response = requests.get(URL, headers=headers, timeout=30)

# Step 9: Stop if the request failed.
response.raise_for_status()

# Step 10: Parse the HTML so we can find elements inside it.
soup = BeautifulSoup(response.text, "lxml")

# Step 11: Locate the main article section.
section = soup.select_one("main div.section")
if section is None:
    raise RuntimeError("Could not find the article section. The page structure may have changed.")

# Step 12: Remove elements that are not part of the real script text.
for unwanted in section.select('script, style, a[href="#qa"], .ecb-publicationDate'):
    unwanted.decompose()

# Step 13: Extract clean text from headings and paragraphs.
text_blocks = []
for element in section.find_all(["h2", "p"]):
    classes = element.get("class", [])

    # Skip the subtitle line with speaker names.
    if "ecb-pressContentSubtitle" in classes:
        continue

    # Pull out plain text from the HTML tag and clean the spacing.
    text = clean_whitespace(element.get_text(" ", strip=True))
    if text:
        text_blocks.append(text)

# Step 14: Join everything into one full document.
full_text = "\n\n".join(text_blocks)

# Step 15: Save the extracted text as a plain text file.
text_path = DATA_DIR / "ecb_press_conference_2026-03-19.txt"
text_path.write_text(full_text, encoding="utf-8")

# Step 16: Download the VADER lexicon and build the sentiment analyzer.
nltk.download("vader_lexicon", quiet=True)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Step 17: Score the full press conference as one document.
scores = sentiment_analyzer.polarity_scores(full_text)
scores["sentiment_label"] = sentiment_label(scores["compound"])
scores["source_url"] = URL

# Step 18: Save the sentiment result as a one-row CSV summary.
sentiment_summary = pd.DataFrame([scores])
sentiment_path = OUTPUT_DIR / "ecb_sentiment_summary.csv"
sentiment_summary.to_csv(sentiment_path, index=False)

# Step 19: Start from the default English stopwords used by the wordcloud package.
custom_stopwords = set(STOPWORDS)

# Step 20: Add domain-specific words that would otherwise dominate the figure.
# These words are common in ECB texts, so removing them helps other themes stand out.
custom_stopwords.update(
    {
        "ecb",
        "euro",
        "area",
        "monetary",
        "policy",
        "inflation",
        "per",
        "cent",
        "will",
        "would",
        "could",
        "also",
        "question",
        "questions",
        "answer",
        "answers",
        "think",
        "going",
    }
)

# Step 21: Tokenize the clean text and remove stopwords.
tokens = tokenize_words(full_text, custom_stopwords)

# Step 22: Count how often each remaining word appears.
word_counts = Counter(tokens)

# Step 23: Save the top 30 words as a CSV table.
top_words = pd.DataFrame(
    word_counts.most_common(30),
    columns=["word", "count"],
)
top_words_path = OUTPUT_DIR / "ecb_top_words.csv"
top_words.to_csv(top_words_path, index=False)

# Step 24: Build the word cloud image from the full text.
# The display settings control the size, colors, and reproducibility of the figure.
wordcloud = WordCloud(
    width=1200,
    height=700,
    background_color="white",
    stopwords=custom_stopwords,
    colormap="viridis",
    random_state=42,
).generate(full_text)

# Step 25: Choose the output path for the word cloud image.
wordcloud_path = OUTPUT_DIR / "ecb_wordcloud.png"

# Step 26: Draw the word cloud with matplotlib and save it as a PNG file.
plt.figure(figsize=(12, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(wordcloud_path, dpi=200, bbox_inches="tight")
plt.close()

# Step 27: Print a short summary so we can confirm all output files were created.
print("Saved text to:", text_path)
print("Saved sentiment summary to:", sentiment_path)
print("Saved top words table to:", top_words_path)
print("Saved word cloud to:", wordcloud_path)
print()
print("Whole-document sentiment scores:")
print(sentiment_summary[["neg", "neu", "pos", "compound", "sentiment_label"]])
print()
print("Top 10 words after stopword removal:")
print(top_words.head(10))
