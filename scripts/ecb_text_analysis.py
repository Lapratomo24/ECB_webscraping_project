from pathlib import Path

import requests

# Step 1: Store the webpage address in a variable so we only need to write it once.
URL = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2026/html/ecb.is260319~93b1cbad97.en.html"

# Step 2: Create folder paths for saved files.
# We make these now even though Step 1 only downloads the page.
# This keeps the project structure consistent from the beginning.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Step 3: Create the folders if they do not already exist.
# exist_ok=True means Python will not crash if the folders are already there.
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 4: Send a browser-like identity to the website.
# Some websites are happier to respond when a request looks like it comes from a real browser.
headers = {
    "User-Agent": "Mozilla/5.0 (beginner text analysis tutorial)"
}

# Step 5: Download the ECB webpage.
# timeout=30 means Python will wait up to 30 seconds before giving up.
response = requests.get(URL, headers=headers, timeout=30)

# Step 6: Stop the script immediately if the website returns an error code.
# For example, 404 means page not found and 500 means server error.
response.raise_for_status()

# Step 7: Print simple checks so we know the download worked.
print("Downloaded page successfully")
print("Status code:", response.status_code)
print("Number of characters in HTML:", len(response.text))
