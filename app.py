import os
import time
import requests
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"

app = Flask(__name__)

# Web Scraping Function
def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # Prevent bot blocking
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise error for bad responses (404, 500, etc.)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:5000]  # Limit text to 5000 chars
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Gemini AI Summarization Function
def summarize_text(raw_text):
    if not GEMINI_API_KEY:
        return ["Error: Missing Gemini API key in .env file."]

    prompt = f"Summarize the following text in bullet points:\n\n{raw_text}"

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )

        # Rate limit handling
        if response.status_code == 429:
            print("[ERROR] Rate limit exceeded. Retrying after 5 seconds...")
            time.sleep(5)
            return summarize_text(raw_text)  # Retry request

        response.raise_for_status()
        summary_data = response.json()

        # Extract summarized bullet points
        if "candidates" in summary_data and summary_data["candidates"]:
            return summary_data["candidates"][0]["content"]["parts"][0]["text"].split("\n")

        return ["Error: Failed to generate summary."]

    except requests.exceptions.RequestException as e:
        return [f"Error: {str(e)}"]

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "Missing URL"}), 400

    raw_data = scrape_website(url)
    if "Error" in raw_data:
        return jsonify({"error": raw_data}), 500

    summary = summarize_text(raw_data)
    return jsonify({"summary": summary})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
