import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
import pickle
import re

# Optional imports
try:
    from PyPDF2 import PdfReader
except:
    PdfReader = None

try:
    import docx
except:
    docx = None

app = Flask(__name__)

# ---------------- MODEL LOADING ----------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    model_loaded = True
except:
    model_loaded = False


# ---------------- TEXT CLEAN ----------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())


# ---------------- FILE READ ----------------
def read_pdf(file):
    if not PdfReader:
        return ""
    text = ""
    try:
        pdf = PdfReader(file)
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + " "
    except:
        return ""
    return text


def read_docx(file):
    if not docx:
        return ""
    try:
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    except:
        return ""


# 🔥 NEW: TXT SUPPORT
def read_txt(file):
    try:
        return file.read().decode("utf-8")
    except:
        return ""


# ---------------- SKILLS ----------------
skills_list = [
    "python", "c", "c++", "java", "kotlin",
    "data analysis", "data science", "machine learning", "deep learning", "nlp",
    "software development", "system design",
    "backend development", "frontend development", "full stack development",
    "html", "css", "javascript", "react",
    "flask", "django", "api development", "web development", "rest apis",
    "sql", "database", "database management",
    "pandas", "data visualization",
    "automation", "testing", "manual testing", "automation testing", "selenium",
    "debugging",
    "docker", "kubernetes", "jenkins", "ci/cd", "linux",
    "aws", "azure", "cloud deployment", "cloud security",
    "deployment", "monitoring", "virtualization",
    "networking", "networking basics", "routing", "switching",
    "socket programming", "hardware support", "troubleshooting",
    "cyber security", "ethical hacking", "security",
    "embedded systems", "microcontrollers",
    "operating systems", "multithreading",
    "robotics", "sensors", "control systems",
    "data structures", "algorithms",
    "excel", "power bi", "business analysis",
    "figma", "wireframing", "prototyping", "user research", "ui design",
    "photoshop", "illustrator", "design principles", "creativity",
    "unity", "game design", "animation",
    "seo", "sem", "social media marketing", "analytics",
    "solidity", "ethereum", "smart contracts", "cryptography",
    "documentation", "writing", "editing", "blogging", "content strategy",
    "product strategy", "communication", "planning", "analysis"
]


# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    text = text.lower()
    return [s for s in skills_list if s in text]


# ---------------- ATS ----------------
def calculate_ats(text):
    score = 0
    words = len(text.split())

    if words > 400:
        score += 30
    elif words > 200:
        score += 20
    else:
        score += 10

    score += len(extract_skills(text)) * 5

    for sec in ["education", "skills", "project", "experience"]:
        if sec in text.lower():
            score += 10

    return min(score, 100)


def resume_strength(score):
    if score < 40:
        return "Weak ❌"
    elif score < 70:
        return "Good 👍"
    else:
        return "Strong 🔥"


# 🔥 FIXED KEYWORD FUNCTION
def keyword_score(text):
    text = text.lower()
    matched = [skill for skill in skills_list if skill in text]
    return min(len(matched) * 5, 100)


# ---------------- SUGGESTIONS ----------------
def get_suggestions(text):
    s = []
    t = text.lower()

    if "project" not in t:
        s.append("Add Projects section")
    if "experience" not in t:
        s.append("Add Experience section")
    if len(extract_skills(text)) < 5:
        s.append("Add more technical skills")
    if len(text.split()) < 200:
        s.append("Increase resume content")

    return s


# ---------------- MAIN ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    data = {
        "prediction": None,
        "ats_score": 0,
        "skills": [],
        "strength": "",
        "keyword": 0,
        "suggestions": [],
        "filename": None,
        "word_count": 0,
        "resume_text": ""
    }

    if request.method == "POST":
        text = ""

        # File input
        file = request.files.get("file")

        if file and file.filename:
            data["filename"] = file.filename
            name = file.filename.lower()

            if name.endswith(".pdf"):
                text += read_pdf(file)
            elif name.endswith(".docx"):
                text += read_docx(file)
            elif name.endswith(".txt"):
                text += read_txt(file)

        if text.strip():
            data["resume_text"] = text
            data["word_count"] = len(text.split())

            # MODEL PREDICTION
            if model_loaded:
                try:
                    cleaned = clean_text(text)
                    vector = vectorizer.transform([cleaned])
                    pred = model.predict(vector)[0]
                    data["prediction"] = le.inverse_transform([pred])[0]
                except:
                    data["prediction"] = "Prediction error"
            else:
                data["prediction"] = "Model not loaded"

            data["ats_score"] = calculate_ats(text)
            data["skills"] = extract_skills(text)
            data["strength"] = resume_strength(data["ats_score"])
            data["keyword"] = keyword_score(text)
            data["suggestions"] = get_suggestions(text)

        else:
            data["prediction"] = "No text found"

    return render_template("index.html", **data)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)