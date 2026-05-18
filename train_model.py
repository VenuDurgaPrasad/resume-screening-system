import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- LOAD DATA ----------------
df = pd.read_csv("resumes.csv")

# 🔥 IMPORTANT FIX (column rename)
if "resume_text" in df.columns:
    df = df.rename(columns={"resume_text": "text"})
elif "Resume" in df.columns:
    df = df.rename(columns={"Resume": "text"})

if "Category" in df.columns:
    df = df.rename(columns={"Category": "category"})

# ---------------- DATA PREPROCESS ----------------
df = df[["text", "category"]]   # only needed columns
df.dropna(inplace=True)

df["text"] = df["text"].apply(clean_text)

# ---------------- ENCODING ----------------
le = LabelEncoder()
df["category"] = le.fit_transform(df["category"])

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42
)

# ---------------- VECTORIZATION ----------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------- ACCURACY ----------------
accuracy = model.score(X_test_vec, y_test)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ---------------- SAVE FILES ----------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("🎯 Model saved successfully!")