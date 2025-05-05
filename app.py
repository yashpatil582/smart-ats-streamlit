import streamlit as st
import ollama
import PyPDF2 as pdf
from docx import Document
import re
import json

# ----------------------------
# Utility Functions
# ----------------------------

MODEL_NAME = "mistral-small3.1"  # centralised model selector

def add_vertical_space(lines: int = 1):
    """Simple vertical spacer that avoids the streamlit‑extras dependency."""
    for _ in range(lines):
        st.markdown("&nbsp;")


def extract_text(uploaded_file):
    """Return plain text from PDF, DOCX, or TXT Streamlit uploaded file."""
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    # PDF
    if filename.endswith(".pdf"):
        reader = pdf.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    # Word (DOCX)
    if filename.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    # Plain‑text fallback
    return uploaded_file.read().decode("utf-8", errors="ignore")


def find_email(text):
    """Return the first email address found in the supplied text, or None."""
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def call_llama(prompt: str) -> str:
    """Send a single‑turn prompt to the local model via Ollama."""
    response = ollama.chat(
        model=MODEL_NAME,  # uses the global model selector
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Smart ATS with Mistral Small 3.1", page_icon="📄")

with st.sidebar:
    st.title("Smart ATS for Resumes (OSS Edition)")
    st.subheader("About")
    st.write(
        f"""
        **Open‑source résumé matcher** leveraging 🦙 **{MODEL_NAME}** locally via **Ollama**.
        
        * Scores your résumé against a job description
        * Surfaces missing keywords
        * Generates a concise profile summary
        """
    )
    add_vertical_space(2)
    st.markdown("---")
    st.write("Made with 💙 by Your Name")

st.title("📑 Résumé ⇄ JD Matcher")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload **Resume**", type=["pdf", "docx", "txt"])
with col2:
    jd_file = st.file_uploader("Upload **Job Description** (optional)", type=["pdf", "docx", "txt"])

jd_textarea = st.text_area("…or paste Job Description here (overrides uploaded JD)", height=200)

if st.button("🚀 Evaluate"):
    # Basic validation
    if resume_file is None and not jd_textarea and jd_file is None:
        st.error("Please upload a resume and a job description or paste one above.")
        st.stop()

    resume_text = extract_text(resume_file) if resume_file else ""
    jd_text = jd_textarea.strip() or extract_text(jd_file)

    if not resume_text or not jd_text:
        st.error("Both résumé and job‑description content are required.")
        st.stop()

    # Construct model prompt
    prompt = (
        "You are an advanced Applicant Tracking System (ATS) specialising in technical roles. "
        "Evaluate the résumé against the job description and respond in **valid JSON** with keys "
        "'JD Match', 'MissingKeywords', and 'Profile Summary'.\n\n"
        f"résumé:\n{resume_text}\n\n"
        f"description:\n{jd_text}\n"
    )

    raw_output = call_llama(prompt)

    # Attempt to parse JSON
    try:
        ats_json = json.loads(raw_output)
    except json.JSONDecodeError:
        st.error("Model returned invalid JSON. Raw output below:")
        st.code(raw_output)
        st.stop()

    # Extract email from résumé text
    email = find_email(resume_text)

    # Top‑level structured summary
    summary = {
        "File": resume_file.name if resume_file else "N/A",
        "Email": email or "Not found",
        "JD Match": ats_json.get("JD Match", "N/A")
    }

    st.subheader("🎯 Quick Results")
    st.json(summary)

    st.subheader("📋 Full ATS Output")
    st.json(ats_json)

    st.success("Analysis complete!")
