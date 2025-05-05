import streamlit as st
import ollama
import PyPDF2 as pdf
from docx import Document
import re
import json

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "gemma:4b-q4_K_M"  # local Ollama tag that fits ~8 GiB RAM

# ----------------------------
# Helper functions
# ----------------------------

def add_vertical_space(lines: int = 1):
    for _ in range(lines):
        st.markdown("&nbsp;")


def extract_text(upload_file):
    if not upload_file:
        return ""
    fname = upload_file.name.lower()
    if fname.endswith(".pdf"):
        reader = pdf.PdfReader(upload_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if fname.endswith(".docx"):
        doc = Document(upload_file)
        return "\n".join(p.text for p in doc.paragraphs)
    return upload_file.read().decode("utf-8", errors="ignore")


def find_email(text: str):
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def chat_llm(prompt: str) -> str:
    try:
        resp = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"].strip()
    except ollama.ResponseError as e:
        if "requires more system memory" in str(e):
            raise RuntimeError("❌ Gemma 4B variant needs more RAM. Try pulling a 2B quant: `ollama pull gemma:2b-q4_K_M`.") from None
        raise


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Smart ATS · Gemma 4B", page_icon="📄")

with st.sidebar:
    st.title("Smart ATS for Resumes")
    st.subheader("Powered by Gemma 4B‑q4_K_M")
    st.write("""
        * Upload **multiple** resumes at once.
        * Scores each résumé vs. the job description.
        * Highlights missing keywords and provides a profile summary.
    """)
    add_vertical_space(2)
    st.markdown("---")
    st.write("Made with 💙 by Your Name")

st.title("📑 Batch Résumé ⇄ JD Matcher")

col1, col2 = st.columns(2)
with col1:
    resume_files = st.file_uploader("Upload **Resumes** (PDF/DOCX/TXT, multiple)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
with col2:
    jd_file = st.file_uploader("Upload **Job Description**", type=["pdf", "docx", "txt"])

jd_textarea = st.text_area("…or paste Job Description here (overrides upload)", height=200)

if st.button("🚀 Evaluate Batch"):
    if not resume_files or (not jd_file and not jd_textarea.strip()):
        st.error("Please upload at least one résumé and provide a job description.")
        st.stop()

    jd_text = jd_textarea.strip() or extract_text(jd_file)

    results = []
    progress = st.progress(0, text="Processing resumes…")

    for idx, res_file in enumerate(resume_files, start=1):
        resume_text = extract_text(res_file)

        prompt = (
            "You are an advanced Applicant Tracking System (ATS) specialising in technical roles. "
            "Evaluate the résumé against the job description and respond in **valid JSON** with keys "
            "'JD Match', 'MissingKeywords', and 'Profile Summary'.\n\n"
            f"résumé:\n{resume_text}\n\n"
            f"description:\n{jd_text}\n"
        )

        try:
            raw = chat_llm(prompt)
            ats = json.loads(raw)
        except (RuntimeError, json.JSONDecodeError) as err:
            st.error(f"Error processing {res_file.name}: {err}")
            if 'raw' in locals():
                st.code(raw)
            continue

        results.append({
            "File": res_file.name,
            "Email": find_email(resume_text) or "Not found",
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

    for res in results:
        with st.expander(f"{res['File']} — Match {res['JD Match']}"):
            st.json(res)

    st.success("Batch analysis complete!")
