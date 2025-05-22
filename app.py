import streamlit as st
import ollama
import PyPDF2 as pdf
from docx import Document
import pandas as pd
import re
import json

# --------------------------------------------------
# Config
# --------------------------------------------------
MODEL_NAME = "llama3.1:8b"  # ensure the model is pulled locally: `ollama pull llama3.1:8b`

DEPARTMENTS = [
    "Marketing & Business Development",
    "Software Development",
    "Project Management",
    "Finance & Accounting",
    "Data Analytics",
]

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def add_vertical_space(n: int = 1):
    for _ in range(n):
        st.markdown("&nbsp;")


def extract_text(f):
    """Extract raw text from PDF / DOCX / TXT FileUploader object."""
    if not f:
        return ""
    name = f.name.lower()
    if name.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in pdf.PdfReader(f).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(f).paragraphs)
    # plain‑text fallback
    return f.read().decode("utf-8", errors="ignore")


def find_email(text: str):
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def chat_llm(prompt: str) -> str:
    """Send single‑turn prompt to local Llama via Ollama and return the raw string response."""
    try:
        resp = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"].strip()
    except ollama.ResponseError as e:
        if "not found" in str(e):
            raise RuntimeError(f"Model '{MODEL_NAME}' not found — pull it with `ollama pull {MODEL_NAME}`")
        raise


def ats_match(resume_text: str, jd_text: str) -> str | int:
    """Return the ATS match percentage (int or 'Error')."""
    prompt = (
        "You are an Applicant Tracking System. Compare the résumé with the job description and respond ONLY "
        "with valid JSON containing exactly one key 'JD Match' whose value is an integer percentage.\n\n"
        f"Résumé:\n{resume_text}\n\nDescription:\n{jd_text}"
    )
    raw = chat_llm(prompt)
    try:
        return json.loads(raw).get("JD Match", "N/A")
    except json.JSONDecodeError:
        return "Error"


def classify_departments(resume_text: str) -> list[str]:
    """Return list of departments a candidate fits best, possibly empty."""
    departments_str = ", ".join(DEPARTMENTS)
    prompt = (
        "As an expert HR screener, choose ALL applicable departments from the following list that best match the "
        "candidate's résumé. Return ONLY valid JSON like {\"Departments\": [\"Software Development\", ...]}. "
        f"If none apply, return an empty list.\n\nDepartments: {departments_str}\n\nRésumé:\n{resume_text}"
    )
    raw = chat_llm(prompt)
    try:
        return json.loads(raw).get("Departments", [])
    except json.JSONDecodeError:
        return []

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="Smart ATS · Llama 3.1 8B", page_icon="📄")

with st.sidebar:
    st.title("Smart ATS for Resumes")
    st.subheader("Powered by Llama 3.1 8B")
    st.markdown("""Upload multiple résumés, compare them to one JD, and get an ATS match percentage plus suggested departments.""")
    add_vertical_space(2)
    st.markdown("---")
    st.caption("Made with 💙 by Your Name")

st.title("📑 Batch Résumé ⇄ JD Matcher")

col1, col2 = st.columns(2)
with col1:
    resume_files = st.file_uploader("Upload **Résumés**", type=["pdf", "docx", "txt"], accept_multiple_files=True)
with col2:
    jd_file = st.file_uploader("Upload **Job Description**", type=["pdf", "docx", "txt"])

jd_textarea = st.text_area("…or paste JD here (overrides upload)", height=200)

if st.button("🚀 Evaluate Batch"):
    if not resume_files or (not jd_file and not jd_textarea.strip()):
        st.error("Need at least one résumé and a job description.")
        st.stop()

    jd_text = jd_textarea.strip() or extract_text(jd_file)
    results = []
    progress = st.progress(0, text="Processing…")

    for idx, rfile in enumerate(resume_files, 1):
        resume_text = extract_text(rfile)
        jd_match_val = ats_match(resume_text, jd_text)
        depts = classify_departments(resume_text)
        dept_display = "; ".join(depts) if depts else "Uncertain"

        results.append({
            "File": rfile.name,
            "Email": find_email(resume_text) or "Not found",
            "JD Match %": jd_match_val,
            "Departments": dept_display,
        })
        progress.progress(idx / len(resume_files))

    progress.empty()

    df = pd.DataFrame(results)
    st.subheader("📊 Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "ats_results.csv", "text/csv")

    st.success("Done!")
