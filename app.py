import streamlit as st
import os
import fitz  # PyMuPDF for PDF parsing
import openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import plotly.graph_objects as go
from serpapi import GoogleSearch
import time

# --- Load API keys ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

# --- Initialize session state ---
for key in ["gap_analysis", "missing_skills", "weekly_plan", "score"]:
    if key not in st.session_state:
        st.session_state[key] = None


# --- Helper functions ---
def extract_text_from_pdf(file):
    text = ""
    if isinstance(file, str):
        doc = fitz.open(file)
    else:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        file.seek(0)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def get_embedding(text, model="text-embedding-3-large"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def compute_similarity(cv_text, jd_text):
    cv_emb = get_embedding(cv_text)
    jd_emb = get_embedding(jd_text)
    return cosine_similarity([cv_emb], [jd_emb])[0][0]


def extract_missing_skills(cv_text, jd_text):
    prompt = f"""
    Compare the following CV and Job Description.
    List the TOP 5 most important skills or requirements that the CV is missing 
    or has only partial coverage for.

    CV: {cv_text[:2000]}
    JD: {jd_text[:2000]}

    Answer ONLY as a comma-separated list of skills.
    """
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()


def search_certifications(skill, max_results=3):
    if not serpapi_key:
        return [{"title": "⚠️ Missing SERPAPI_API_KEY", "link": "", "snippet": ""}]

    params = {
        "engine": "google",
        "q": f"best certification for {skill} site:coursera.org OR site:udemy.com OR site:edx.org OR site:linkedin.com",
        "num": max_results,
        "api_key": serpapi_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    certs = []
    if "organic_results" in results:
        for res in results["organic_results"][:max_results]:
            certs.append({
                "title": res.get("title"),
                "link": res.get("link"),
                "snippet": res.get("snippet")
            })
    return certs


def generate_weekly_plan(missing_skills, months):
    all_certs = {}
    for skill in missing_skills.split(","):
        skill = skill.strip()
        all_certs[skill] = search_certifications(skill)

    prompt = f"""
    The student has {months} months until placements.
    Missing skills: {missing_skills}.
    Below are real certifications from Coursera, edX, Udemy, and LinkedIn Learning:

    {all_certs}

    Based on these, recommend the best ones and create a weekly learning plan.
    Format:
    Week | Focus Area | Certification (with provider + link) | Milestone
    """
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content


def generate_gap_analysis(cv_text, jd_text, score):
    prompt = f"""
    Provide a structured gap analysis between this CV and the Job Description.
    Include:
    - Strengths
    - Missing or weak skills
    - Suggestions to improve CV
    Format clearly with bullet points.

    CV: {cv_text[:2000]}
    JD: {jd_text[:2000]}
    Match Score: {round(score*100,2)}%
    """
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content


def load_companies_and_roles(jd_folder="jds"):
    companies = {}
    for fname in os.listdir(jd_folder):
        if fname.endswith(".pdf"):
            try:
                company, role = fname.replace(".pdf", "").split("_", 1)
                if company not in companies:
                    companies[company] = []
                companies[company].append(role)
            except ValueError:
                pass
    return companies


# --- Streamlit UI ---
st.set_page_config(page_title="MBA Skill Gap Analyzer", layout="wide")

st.title("🎓 MBA Student Skill Gap Analyzer (Prototype)")

# Sidebar controls
with st.sidebar:
    st.header("Upload & Select Options")
    cv_file = st.file_uploader("📄 Upload your CV (PDF)", type=["pdf"])

    companies = load_companies_and_roles("jds")

    if companies:
        company = st.selectbox("🏢 Select Company", list(companies.keys()))
        program = st.radio("🎓 Select Program", ["PGDM", "PGDM DCP", "PGDM BFSI", "PGDM Exp"])
        role = st.selectbox("💼 Select Role", companies[company])
    else:
        st.error("No JD PDFs found in /jds folder. Please add them first.")
        company, role = None, None

    months = st.number_input("⏳ Months until placement", min_value=1, max_value=12, step=1)

    analyze_btn = st.button("🔍 Analyze CV")
    plan_btn = st.button("📝 Generate Plan")


# --- Processing Logic ---
if analyze_btn and cv_file and company and role:
    with st.spinner("Analyzing CV vs JD... ⏳"):
        time.sleep(1)
        cv_text = extract_text_from_pdf(cv_file)
        jd_file = f"jds/{company}_{role}.pdf"

        if os.path.exists(jd_file):
            jd_text = extract_text_from_pdf(jd_file)

            score = compute_similarity(cv_text, jd_text)
            st.session_state.score = round(score * 100, 2)

            st.session_state.gap_analysis = generate_gap_analysis(cv_text, jd_text, score)
            st.session_state.missing_skills = extract_missing_skills(cv_text, jd_text)
            st.session_state.weekly_plan = None  # reset old plan
        else:
            st.error(f"JD file not found: {jd_file}")


if plan_btn and st.session_state.missing_skills:
    with st.spinner("Generating personalized learning plan... 📝"):
        time.sleep(1)
        st.session_state.weekly_plan = generate_weekly_plan(
            st.session_state.missing_skills, months
        )


# --- Results Area ---
if st.session_state.gap_analysis:
    col1, col2 = st.columns([1, 2])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state.score,
            title={'text': "Match Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': "#ff4d4d"},
                    {'range': [50, 75], 'color': "#ffcc00"},
                    {'range': [75, 100], 'color': "#66cc66"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Results")
        tab1, tab2, tab3 = st.tabs(["Gap Analysis", "Missing Skills", "Learning Plan"])

        with tab1:
            with st.expander("View Detailed Gap Analysis"):
                st.write(st.session_state.gap_analysis)

        with tab2:
            st.write("### Missing Skills")
            st.write(st.session_state.missing_skills)

        with tab3:
            if st.session_state.weekly_plan:
                st.write("### Weekly Learning Plan")
                st.write(st.session_state.weekly_plan)
            else:
                st.info("Click **Generate Plan** in the sidebar to get your learning plan.")
