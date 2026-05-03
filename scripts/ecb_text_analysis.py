from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup


# Step 1: Store the webpage address in one variable.
URL = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2026/html/ecb.is260319~93b1cbad97.en.html"

# Step 2: Define where saved files should go.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Step 3: Create those folders if needed.
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_whitespace(text: str) -> str:
    # Step 4: Clean messy spacing.
    # HTML text can contain extra spaces, tabs, and line breaks.
    # This function turns repeated whitespace into a single clean space.
    """Turn repeated spaces, tabs, and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# Step 5: Identify our request as coming from a browser-like script.
headers = {
    "User-Agent": "Mozilla/5.0 (beginner text analysis tutorial)"
}

# Step 6: Download the webpage HTML.
response = requests.get(URL, headers=headers, timeout=30)

# Step 7: Crash early if the download failed.
response.raise_for_status()

# Step 8: Parse the HTML with BeautifulSoup so we can search inside it.
# "lxml" is the parser engine. It is fast and commonly used.
soup = BeautifulSoup(response.text, "lxml")

# Step 9: Find the main content block that holds the press conference text.
# We inspected the page in the browser and found that the useful content is in main div.section.
section = soup.select_one("main div.section")
if section is None:
    raise RuntimeError("Could not find the article section. The page structure may have changed.")

# Step 10: Remove page elements we do not want in the analysis text.
# This avoids mixing navigation or page metadata into the script text.
for unwanted in section.select('script, style, a[href="#qa"], .ecb-publicationDate'):
    unwanted.decompose()

# Step 11: Prepare an empty list to collect each cleaned heading or paragraph.
text_blocks = []

# Step 12: Loop through headings and paragraphs inside the section.
# We use both h2 and p because the section titles are meaningful text too.
for element in section.find_all(["h2", "p"]):
    classes = element.get("class", [])

    # Step 13: Skip the subtitle line with speaker names.
    # That line is useful on the webpage but not central to the analysis text.
    if "ecb-pressContentSubtitle" in classes:
        continue

    # Step 14: Extract visible text from the HTML tag and clean its spacing.
    text = clean_whitespace(element.get_text(" ", strip=True))

    # Step 15: Only keep non-empty text blocks.
    if text:
        text_blocks.append(text)

# Step 16: Combine all text blocks into one long script.
# We put blank lines between blocks to keep the text readable.
full_text = "\n\n".join(text_blocks)

# Step 17: Choose the output filename for the cleaned text.
text_path = DATA_DIR / "ecb_press_conference_2026-03-19.txt"

# Step 18: Save the full script as a UTF-8 text file.
text_path.write_text(full_text, encoding="utf-8")

# Step 19: Print a small report so we can check what happened.
print("Saved text to:", text_path)
print("Number of extracted text blocks:", len(text_blocks))
print()
print("Preview:")
print(full_text[:800])
