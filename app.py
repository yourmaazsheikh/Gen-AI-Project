from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

app = Flask(__name__)
app.secret_key = "student-performance-demo-key"

model = joblib.load(os.path.join(BASE_DIR, "model", "student_performance_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

# Order matters: must match the order used in train_model.py
FEATURES = [
    "attendance", "study_hours", "previous_score", "assignments",
    "participation", "sleep_hours", "extracurricular",
]


# ---------------------------------------------------------------- database
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attendance REAL NOT NULL,
                study_hours REAL NOT NULL,
                previous_score REAL NOT NULL,
                assignments REAL NOT NULL,
                participation REAL NOT NULL,
                sleep_hours REAL NOT NULL,
                extracurricular REAL NOT NULL,
                result REAL NOT NULL,
                label TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def performance_label(score):
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Improvement"


def get_record(record_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM predictions WHERE id = ?", (record_id,)
        ).fetchone()


def get_previous_record(record_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM predictions WHERE id < ? ORDER BY id DESC LIMIT 1",
            (record_id,),
        ).fetchone()


def get_latest_record():
    with get_db() as db:
        return db.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 1"
        ).fetchone()


def get_history(limit=15):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


# ------------------------------------------------------------------ routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]
        except (ValueError, KeyError):
            # keep what the user typed so nothing is lost
            flash("Please enter valid numeric values in all fields.")
            return render_template(
                "index.html",
                form=request.form, record=None, delta=None,
                history=get_history(), features=FEATURES,
            )

        prediction = float(model.predict(scaler.transform([values]))[0])
        result = round(max(0, min(100, prediction)), 2)
        label = performance_label(result)

        with get_db() as db:
            cur = db.execute(
                f"""INSERT INTO predictions ({", ".join(FEATURES)}, result, label)
                    VALUES ({", ".join("?" * len(FEATURES))}, ?, ?)""",
                (*values, result, label),
            )
            new_id = cur.lastrowid

        # Post/Redirect/Get: avoids re-submitting on refresh
        return redirect(url_for("index", record=new_id))

    # GET: show a chosen record, otherwise the most recent one
    record_id = request.args.get("record", type=int)
    record = get_record(record_id) if record_id else get_latest_record()

    delta = None
    if record:
        prev = get_previous_record(record["id"])
        if prev:
            delta = round(record["result"] - prev["result"], 2)

    return render_template(
        "index.html",
        form=record, record=record, delta=delta,
        history=get_history(), features=FEATURES,
    )


@app.route("/new")
def new_entry():
    """Blank form (history stays in the database)."""
    return render_template(
        "index.html", form=None, record=None, delta=None,
        history=get_history(), features=FEATURES,
    )


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete(record_id):
    with get_db() as db:
        db.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
    return redirect(url_for("index"))


@app.route("/clear", methods=["POST"])
def clear_history():
    with get_db() as db:
        db.execute("DELETE FROM predictions")
    return redirect(url_for("new_entry"))


init_db()

if __name__ == "__main__":
    app.run(debug=True)
