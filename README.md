# AI-Driven Student Performance Prediction System

This project predicts a student's expected final academic score using Machine Learning.

## Tech Stack
- Python
- Flask
- NumPy / Pandas
- Scikit-learn
- Random Forest Regression
- SQLite (built into Python) for saving prediction history
- HTML / CSS

## Model
Random Forest Regressor is trained on 600 synthetic student records.

Test-set metrics from the included training run:
- MAE: 3.24
- R²: 0.80

## Project Structure

AI_Student_Performance_Prediction/
├── app.py
├── train_model.py
├── database.db        (auto-created on first run)
├── requirements.txt
├── README.md
├── data/
│   └── student_data.csv
├── model/
│   ├── student_performance_model.pkl
│   └── scaler.pkl
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── images/ai_background.svg

## Run on Windows

Open CMD/PowerShell in this folder:

1. Create environment:
   python -m venv venv

2. Activate:
   venv\Scripts\activate

3. Install:
   pip install -r requirements.txt

4. If you modify the CSV, retrain:
   python train_model.py

5. Start:
   python app.py

6. Open:
   http://127.0.0.1:5000

## Features
- AI/neural-network themed background with glass-style cards
- Every prediction is saved to SQLite (`database.db`)
- After predicting, the form keeps your inputs: change any one field and predict again
- "Previous Records" table: Load any old record back into the form, or delete it
- Shows the % change compared with the previous record

## Note
The dataset is synthetic/demo data for an academic project. It should not be used to make real educational decisions without validated real-world data and appropriate privacy/ethical safeguards.
