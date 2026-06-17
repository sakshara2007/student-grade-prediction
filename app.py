import pickle
import sqlite3
import datetime
import numpy as np

from flask import Flask, render_template, request, jsonify, g

app = Flask(__name__)

DB_PATH = "predictions.db"

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
with open("best_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

with open("models/features.pkl", "rb") as f:
    FEATURES = pickle.load(f)

with open("models/metrics.pkl", "rb") as f:
    METRICS = pickle.load(f)


# -----------------------------
# DATABASE
# -----------------------------
def get_db():
    db = getattr(g, "_database", None)

    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row

    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)

    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            timestamp TEXT,
            predicted REAL,
            grade_band TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# -----------------------------
# GRADE BAND
# -----------------------------
def grade_band(score):

    if score >= 16:
        return "Excellent"

    elif score >= 12:
        return "Good"

    elif score >= 10:
        return "Average"

    else:
        return "Needs Improvement"


# -----------------------------
# STUDY TIPS
# -----------------------------
def generate_tips(data, predicted):

    studytime = int(data.get("studytime", 2))
    failures = int(data.get("failures", 0))
    absences = int(data.get("absences", 0))
    goout = int(data.get("goout", 3))
    health = int(data.get("health", 3))
    freetime = int(data.get("freetime", 3))

    tips = []

    if studytime < 3:
        tips.append("Increase your weekly study hours to improve academic performance.")

    if failures > 0:
        tips.append("Focus on strengthening weak subjects through regular practice and revision.")

    if absences > 5:
        tips.append("Maintaining consistent attendance can positively impact your grades.")

    if goout > 3:
        tips.append("Balancing social activities with study time can improve learning outcomes.")

    if health < 3:
        tips.append("Good sleep, exercise, and overall wellbeing support better academic results.")

    if freetime > 4:
        tips.append("Consider using part of your free time for revision and skill development.")

    if predicted < 10:
        tips.append("Additional academic support may help improve future performance.")

    defaults = [
        "Create a weekly study schedule and follow it consistently.",
        "Practice active recall and solve problems regularly.",
        "Review class notes daily instead of studying only before exams."
    ]

    combined = list(dict.fromkeys(tips + defaults))

    return combined[:3]


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# PREDICT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    form = request.form
    student_name = form.get("student_name")

    row = {f: 0 for f in FEATURES}

    fields = [
        "studytime",
        "failures",
        "absences",
        "goout",
        "health",
        "freetime"
    ]

    for field in fields:
        try:
            row[field] = int(form.get(field, 0))
        except:
            row[field] = 0

    X = np.array([[row.get(f, 0) for f in FEATURES]])

    prediction = MODEL.predict(
        SCALER.transform(X)
    )[0]

    predicted = round(float(prediction), 2)

    if predicted < 0:
        predicted = 0

    if predicted > 20:
        predicted = 20

    band = grade_band(predicted)

    tips = generate_tips(form, predicted)

    db = get_db()

    db.execute(
    """
    INSERT INTO predictions
    (student_name,timestamp,predicted,grade_band)
    VALUES (?,?,?,?)
    """,
    (
        student_name,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        predicted,
        band
    )
)

    db.commit()

    return render_template(
        "result.html",
        predicted=predicted,
        band=band,
        tips=tips,
        metrics=METRICS  # <-- Add this line
    )
    


# -----------------------------
# HISTORY API
# -----------------------------
@app.route("/history")
def history():

    rows = get_db().execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 20"
    ).fetchall()

    return jsonify([dict(r) for r in rows])


# -----------------------------
# HISTORY PAGE
# -----------------------------
@app.route("/history_page")
def history_page():

    rows = get_db().execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    ).fetchall()

    return render_template(
        "history.html",
        rows=rows
    )


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)