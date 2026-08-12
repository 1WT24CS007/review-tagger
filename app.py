from flask import Flask, render_template, request, jsonify
import joblib
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

model = joblib.load("models/sentiment_model.pkl")
DATABASE = "data/review_history.db"

ASPECTS = {
    "Battery": ["battery", "charge", "charging", "drains"],
    "Camera": ["camera", "photo", "picture"],
    "Comfort": ["comfort", "comfortable"],
    "Fit": ["fit", "size", "small", "large"],
    "Quality": ["quality", "durable", "broken"],
    "Price": ["price", "cost", "expensive", "cheap"],
    "Design": ["design", "color", "look"],
    "Delivery": ["delivery", "shipping", "arrived"]
}

def create_database():
    os.makedirs("data", exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT,
            sentiment TEXT,
            confidence REAL,
            aspects TEXT,
            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()

create_database()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    review = request.json.get("review", "").lower().strip()

    if not review:
        return jsonify({"error": "Please enter a review first."}), 400

    found_aspects = [
        aspect for aspect, words in ASPECTS.items()
        if any(word in review for word in words)
    ]

    sentiment = model.predict([review])[0]
    probabilities = model.predict_proba([review])[0]
    confidence = round(max(probabilities) * 100, 1)

    aspects_text = ", ".join(found_aspects) if found_aspects else "No specific aspect found"

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO review_history
        (review, sentiment, confidence, aspects, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        review,
        sentiment,
        confidence,
        aspects_text,
        datetime.now().strftime("%d-%m-%Y %H:%M")
    ))

    connection.commit()
    connection.close()

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence,
        "aspects": found_aspects or ["No specific aspect found"]
    })

@app.route("/history")
def history():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        SELECT review, sentiment, confidence, aspects, created_at
        FROM review_history
        ORDER BY id DESC
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    connection.close()

    return jsonify(rows)

if __name__ == "__main__":
    app.run(debug=True)