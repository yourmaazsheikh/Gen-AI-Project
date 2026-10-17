import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

DATA_PATH = "data/student_data.csv"
MODEL_DIR = "model"

df = pd.read_csv(DATA_PATH)

features = [
    "attendance", "study_hours", "previous_score", "assignments",
    "participation", "sleep_hours", "extracurricular"
]

X = df[features]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=250,
    max_depth=12,
    random_state=42
)

model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("Model trained successfully!")
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 :", round(r2_score(y_test, predictions), 2))

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "student_performance_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("Model files saved in model/ folder.")
