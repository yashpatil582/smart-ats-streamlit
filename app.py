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
MODEL_NAME = "llama3.1:8b"       # ensure model is pulled: ollama pull llama3.1:8b

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def add_vertical_space(n: int = 1):
    for _ in range(n):
        st.markdown("&nbsp;")


def extract_text(f):
    if not f:
        return ""
    name = f.name.lower()
    if name.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in pdf.PdfReader(f).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(f).paragraphs)
    return f.read().decode("utf-8", errors="ignore")


def find_email(t: str):
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
    return m.group(0) if m else None


def chat_llm(prompt: str) -> str:
    try:
        r = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return r["message"]["content"].strip()
    except ollama.ResponseError as e:
        if "not found" in str(e):
            raise RuntimeError(f"Model '{MODEL_NAME}' not found — pull it with `ollama pull {MODEL_NAME}`")
        raise


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="Smart ATS · Llama 3.1 8B", page_icon="📄")

with st.sidebar:
    st.title("Smart ATS for Resumes")
    st.subheader("Powered by Llama 3.1 8B")
    st.markdown("""Upload multiple resumes, compare them to one JD, and get an ATS match percentage for each.""")
    add_vertical_space(2)
    st.markdown("---")
    st.caption("Made with 💙 by Your Name")

st.title("📑 Batch Résumé ⇄ JD Matcher")

col1, col2 = st.columns(2)
with col1:
    resume_files = st.file_uploader("Upload **Resumes**", type=["pdf", "docx", "txt"], accept_multiple_files=True)
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
        prompt = (
            "You are an ATS. Evaluate the résumé vs. the job description and respond ONLY with valid JSON "
            "containing the single key 'JD Match' whose value is a percentage (e.g., 83). Do NOT include other keys."\
            "\nRésumé:\n" + resume_text + "\n\nDescription:\n" + jd_text
        )
        try:
            raw = chat_llm(prompt)
            jd_match = json.loads(raw).get("JD Match", "N/A")
        except Exception as err:
            st.error(f"{rfile.name}: {err}")
            jd_match = "Error"

        results.append({
            "File": rfile.name,
            "Email": find_email(resume_text) or "Not found",
            "JD Match %": jd_match,
        })
        progress.progress(idx / len(resume_files))

    progress.empty()

    df = pd.DataFrame(results)
    st.subheader("📊 Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "ats_results.csv", "text/csv")

    st.success("Done!")
