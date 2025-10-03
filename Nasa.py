import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("exoplanet.csv")

print("Shape of dataset:", df.shape)
print("Columns:", df.columns)

drop_cols = ["rowid", "pl_name", "hostname", "discoverymethod"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

df = df.dropna()

if "disposition" in df.columns:
    df["label"] = df["disposition"].apply(lambda x: 1 if x == "CONFIRMED" else 0)
elif "koi_disposition" in df.columns:
    df["label"] = df["koi_disposition"].apply(lambda x: 1 if x == "CONFIRMED" else 0)
else:
    raise ValueError("No disposition column found in dataset!")

X = df.drop(columns=["label", "disposition"], errors="ignore")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Planet Detection")
plt.show()
