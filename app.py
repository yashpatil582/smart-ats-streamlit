import streamlit as st
import ollama
import PyPDF2 as pdf
from docx import Document
import re
import json

# --------------------------------------------------
# Config
# --------------------------------------------------
# Use the 8‑billion‑parameter Llama 3.1 model (quantised by Ollama).
# Common tag:  "llama3.1:8b"  (check with `ollama search llama3.1`).
MODEL_NAME = "llama3.1:8b"

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
        reader = pdf.PdfReader(f)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if name.endswith(".docx"):
        doc = Document(f)
        return "\n".join(p.text for p in doc.paragraphs)
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
            raise RuntimeError(
                f"❌ Model '{MODEL_NAME}' not found. Pull it first: `ollama pull {MODEL_NAME}`."
            ) from None
        if "requires more system memory" in str(e):
            raise RuntimeError(
                "❌ Llama 3.1 8B needs more RAM than available. Consider using a 4‑bit quant (e.g. `llama3.1:8b-q4_K_M`)."
            ) from None
        raise


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="Smart ATS · Llama 3.1 8B", page_icon="📄")

with st.sidebar:
    st.title("Smart ATS for Resumes")
    st.subheader("Powered by Llama 3.1 8B")
    st.write(
        """
        • Upload **multiple** resumes at once  
        • Scores each résumé against the job description  
        • Highlights missing keywords & provides a profile summary
        """
    )
    add_vertical_space(2)
    st.markdown("---")
    st.write("Made with 💙 by Your Name")

st.title("📑 Batch Résumé ⇄ JD Matcher")

c1, c2 = st.columns(2)
with c1:
    resume_files = st.file_uploader(
        "Upload **Resumes** (PDF/DOCX/TXT, multiple)",
        type=["pdf", "docx", "txt"], accept_multiple_files=True,
    )
with c2:
    jd_file = st.file_uploader("Upload **Job Description**", type=["pdf", "docx", "txt"])

jd_textarea = st.text_area("…or paste Job Description here (overrides upload)", height=200)

if st.button("🚀 Evaluate Batch"):
    if not resume_files or (not jd_file and not jd_textarea.strip()):
        st.error("Please upload at least one résumé and provide a job description.")
        st.stop()

    jd_text = jd_textarea.strip() or extract_text(jd_file)

    results = []
    progress = st.progress(0, text="Processing resumes…")

    for idx, r_file in enumerate(resume_files, start=1):
        res_text = extract_text(r_file)
        prompt = (
            "You are an advanced Applicant Tracking System (ATS) specialising in technical roles. "
            "Evaluate the résumé against the job description and respond in **valid JSON** with keys "
            "'JD Match', 'MissingKeywords', and 'Profile Summary'.\n\n"
            f"résumé:\n{res_text}\n\n"
            f"description:\n{jd_text}\n"
        )
        try:
            raw = chat_llm(prompt)
            ats = json.loads(raw)
        except (RuntimeError, json.JSONDecodeError) as err:
            st.error(f"Error processing {r_file.name}: {err}")
            if "raw" in locals():
                st.code(raw)
            continue
        results.append({
            "File": r_file.name,
            "Email": find_email(res_text) or "Not found",
            "JD Match": ats.get("JD Match", "N/A"),
            "MissingKeywords": ats.get("MissingKeywords", []),
            "Profile Summary": ats.get("Profile Summary", "")
        })
        progress.progress(idx / len(resume_files))

    progress.empty()

    if not results:
        st.warning("No resumes processed successfully.")
        st.stop()

    st.subheader("📊 Batch Results")
    for r in results:
        with st.expander(f"{r['File']} — Match {r['JD Match']}"):
            st.json(r)
    st.success("Batch analysis complete!")
