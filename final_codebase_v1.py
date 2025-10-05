# app_streamlit.py
"""
MBA Student Skill Gap Analyzer (with LangChain + Chroma RAG + filename parsing)
This file preserves your UI and adds:
 - Chroma-based RAG retriever (LangChain) to retrieve JD passages
 - Robust filename parsing to populate Company & Role dropdowns
 - A sidebar control to (re)build the Chroma collection from PDFs in /jds

Prereqs:
 - Put JD PDFs in ./jds (filenames like "Acme - Business Analyst.pdf")
 - .env with OPENAI_API_KEY (and SERPAPI_KEY if you use SerpAPI)
 - pip install langchain chromadb openai streamlit python-dotenv pdfminer.six numpy scikit-learn plotly pymupdf serpapi
"""

import os
import json
import time
from io import BytesIO
from typing import List, Dict, Any
import re

import streamlit as st
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text_to_fp
import fitz  # PyMuPDF for alternate extraction (used in older code)
import plotly.graph_objects as go

# LangChain / Chroma imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.retrievers import DocumentRetriever

# OpenAI API (for any direct calls left)
from openai import OpenAI

# SerpAPI (if used by other parts of your code)
from serpapi import GoogleSearch
import requests

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in .env. Add it and restart.")
    st.stop()

# OpenAI client (new)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Config
JDS_FOLDER = "jds"
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "jds_collection"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3

# ----------------------------
# Utilities: PDF -> text
# ----------------------------
def extract_text_from_pdf_bytes(b: bytes) -> str:
    out = BytesIO()
    try:
        extract_text_to_fp(BytesIO(b), out, output_type="text")
        return out.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        # fallback to PyMuPDF if pdfminer fails
        try:
            doc = fitz.open(stream=b, filetype="pdf")
            txt = ""
            for page in doc:
                txt += page.get_text()
            doc.close()
            return txt
        except Exception:
            return ""

def extract_text_from_pdf_file(path: str) -> str:
    with open(path, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())

# ----------------------------
# Robust filename parsing
# ----------------------------
def parse_company_role_from_filename(fname: str) -> Dict[str, str]:
    """
    Parse filenames:
      - "Acme - Business Analyst.pdf"
      - "Acme_BusinessAnalyst.pdf"
      - "Acme - Business Analyst - Senior.pdf"
    Returns dict {"company":..., "role":...}
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    # unify separators
    parts = re.split(r'\s*[-_]\s*', base)
    if len(parts) >= 2:
        company = parts[0].strip()
        # join remaining as role (preserve inner hyphens if any)
        role = " - ".join(p.strip() for p in parts[1:])
        return {"company": company, "role": role}
    # fallback: try splitting by first space
    tokens = base.split()
    if len(tokens) >= 2:
        company = tokens[0].strip()
        role = " ".join(tokens[1:]).strip()
        return {"company": company, "role": role}
    return {"company": base, "role": ""}

def load_companies_and_roles_from_jds(jd_folder: str = JDS_FOLDER) -> Dict[str, List[str]]:
    """Scan the jds folder and return a mapping {company: [roles]} using filename parsing."""
    mapping = {}
    if not os.path.exists(jd_folder):
        return mapping
    for fname in sorted(os.listdir(jd_folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        parsed = parse_company_role_from_filename(fname)
        company = parsed.get("company") or "Unknown"
        role = parsed.get("role") or "General"
        mapping.setdefault(company, set()).add(role)
    # convert to sorted lists
    return {c: sorted(list(roles)) for c, roles in mapping.items()}

# ----------------------------
# Chroma RAG utilities
# ----------------------------
def build_chroma_from_jds(jd_folder: str = JDS_FOLDER, persist_directory: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
    """
    Build (or update) a Chroma collection from PDF files in jd_folder.
    This will read PDFs, create Documents and call Chroma.from_documents to embed & persist.
    WARNING: This will call OpenAI embeddings (cost).
    """
    if not os.path.exists(jd_folder):
        st.warning(f"JD folder '{jd_folder}' not found.")
        return None

    docs: List[Document] = []
    for fname in sorted(os.listdir(jd_folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(jd_folder, fname)
        try:
            text = extract_text_from_pdf_file(path)
        except Exception as e:
            st.warning(f"Failed to read {fname}: {e}")
            continue
        # keep a reasonably sized chunk per doc; you can chunk further if needed
        metadata = {"source": fname}
        docs.append(Document(page_content=text, metadata=metadata))

    if not docs:
        st.warning("No JD documents to ingest.")
        return None

    # Use LangChain OpenAIEmbeddings wrapper (reads OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    chroma_db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory, collection_name=collection_name)
    chroma_db.persist()
    return chroma_db

def load_chroma_store(persist_directory: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
    """Load existing Chroma collection if present."""
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        store = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embeddings)
        return store
    except Exception:
        return None

def rag_retrieve(store: Chroma, query: str, k: int = TOP_K):
    """Retrieve top-k docs using Chroma retriever."""
    if store is None:
        return []
    retriever = store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    # convert to list of dicts
    results = []
    for d in docs:
        results.append({"text": d.page_content, "metadata": d.metadata})
    return results

# ----------------------------
# Existing helper functions (similarity, LLM wrapper)
# ----------------------------
def get_embedding_via_openai(text: str):
    """Use new OpenAI client to produce an embedding (if needed)."""
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=text)
    return np.array(resp.data[0].embedding, dtype=float)

def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(cosine_similarity([a], [b])[0][0])

def call_gap_analysis_llm(cv_text: str, role_text: str, program: str) -> Dict[str, Any]:
    prompt = f"""
You are an expert MBA career coach and skills analyst. Return ONLY valid JSON.

Inputs:
CV_TEXT: {cv_text[:3500]}
ROLE_TEXT: {role_text[:3000]}
PROGRAM: {program}

Tasks:
1) Provide candidate_profile (name if found, emails[], top_skills[]).
2) Extract role_skills from ROLE_TEXT with importance (0-1).
3) Compute match_score (0-100) and one-line explanation.
4) Identify top 1-2 skill gaps with rationale.
5) For each gap suggest up to 3 certification/course options with provider, title, url, estimated_duration_weeks.

Return JSON exactly matching:
{{"candidate_profile":{{}}, "role_skills":[], "match_score":0, "match_explain":"", "gaps":[], "cert_recommendations":[]}}
"""
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert MBA career coach and return only JSON when asked."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=800,
    )
    raw = ""
    try:
        raw = resp.choices[0].message["content"]
    except Exception:
        raw = str(resp)
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1:
        raw_json = raw[first:last+1]
        try:
            return json.loads(raw_json)
        except Exception:
            return {"error_raw": raw}
    return {"error_raw": raw}

# ----------------------------
# Existing search helper (SerpAPI or Google CSE could be used)
# ----------------------------
def search_certs_google(skill: str, num_results: int = 5):
    """Search using Google Custom Search (if configured) else return empty."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    q = f"{skill} certification site:coursera.org OR site:edx.org OR site:udemy.com OR site:linkedin.com"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": q, "num": min(num_results, 10)}
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("items", [])[:num_results]:
            items.append({"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet", "")})
        return items
    except Exception as e:
        st.warning(f"Google CSE search failed: {e}")
        return []

def search_certs_serpapi(skill: str, num_results: int = 5):
    if not SERPAPI_KEY:
        return []
    params = {"engine":"google","q":f"{skill} certification site:coursera.org OR site:edx.org OR site:udemy.com","api_key":SERPAPI_KEY,"num":num_results}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        out = []
        for r in results.get("organic_results", [])[:num_results]:
            out.append({"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet")})
        return out
    except Exception as e:
        st.warning(f"SerpAPI search failed: {e}")
        return []

# ----------------------------
# UI & app logic (preserve your layout)
# ----------------------------
st.set_page_config(page_title="MBA Skill Gap Analyzer", layout="wide")
st.title("🎓 MBA Student Skill Gap Analyzer (Prototype)")

# Sidebar
with st.sidebar:
    st.header("Upload & Select Options")
    cv_file = st.file_uploader("📄 Upload your CV (PDF)", type=["pdf"])
    st.markdown("---")
    st.write("JD source: ./jds (place PDF files there)")

    # Build / load chroma index controls
    st.markdown("### RAG (Chroma) Index")
    if st.button("Build / Rebuild Chroma index from jds"):
        with st.spinner("Building Chroma index — this will call OpenAI embeddings (may cost API credits)..."):
            chroma_store = build_chroma_from_jds(JDS_FOLDER, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)
            if chroma_store:
                st.success("Chroma index built and saved.")
            else:
                st.error("Failed to build Chroma index. Check logs.")

    st.markdown("---")
    # Load company/role mapping from filenames
    companies = load_companies_and_roles_from_jds(JDS_FOLDER)
    if companies:
        company = st.selectbox("🏢 Select Company", [""] + list(companies.keys()))
        program = st.radio("🎓 Select Program", ["PGDM", "PGDM DCP", "PGDM BFSI", "PGDM Exp"])
        roles = companies.get(company, []) if company else []
        role = st.selectbox("💼 Select Role", [""] + roles)
    else:
        st.error("No JD PDFs found in /jds folder. Please add them first.")
        company = None
        role = None

    st.markdown("---")
    st.write("OpenAI key: set in .env")
    if SERPAPI_KEY:
        st.write("SerpAPI configured")
    else:
        st.write("SerpAPI not configured (optional)")

# Initialize session_state placeholders
if "gap_analysis" not in st.session_state:
    st.session_state["gap_analysis"] = None
if "missing_skills" not in st.session_state:
    st.session_state["missing_skills"] = None
if "weekly_plan" not in st.session_state:
    st.session_state["weekly_plan"] = None
if "score" not in st.session_state:
    st.session_state["score"] = None

# Main area: buttons and functionality
col1, col2 = st.columns([1, 3])
with col1:
    analyze_btn = st.button("🔍 Analyze CV")
    plan_btn = st.button("📝 Generate Plan")
with col2:
    st.write("Use the sidebar to upload CV and select Company & Role. Click Analyze to run the agentic flow.")

# Try to load an existing Chroma store for RAG retrieval (if present)
chroma_store = load_chroma_store(CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)

# --- Analyze flow ---
if analyze_btn and cv_file and company and role:
    with st.spinner("Analyzing CV vs JD... ⏳"):
        try:
            # 1) Read CV text
            cv_bytes = cv_file.read()
            cv_text = extract_text_from_pdf_bytes(cv_bytes)

            # 2) Build / use chroma store
            if chroma_store is None:
                # try to build silently (only if files present)
                chroma_store = build_chroma_from_jds(JDS_FOLDER, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)

            # 3) Filter JD candidates by company & role by scanning jds filenames (we keep UI consistent)
            # We'll still perform RAG retrieval across the whole collection but present ones matching company/role
            # Build a company-specific query to bias retriever
            query_for_retriever = f"Relevant JD passages for role '{role}' at company '{company}'."

            # 4) Retrieve top-k JD passages via chroma retriever
            if chroma_store:
                retrieved = rag_retrieve(chroma_store, query_for_retriever, k=TOP_K*2)  # fetch more then filter
            else:
                retrieved = []

            # Filter retrieved results to only those whose metadata source filename matches company & role
            filtered = []
            for d in retrieved:
                src = d.get("metadata", {}).get("source", "")
                parsed = parse_company_role_from_filename(src)
                if parsed.get("company","").lower() == company.lower():
                    # if role is specified, match role (case-insensitive substring)
                    if role:
                        if role.lower() in parsed.get("role","").lower():
                            filtered.append(d)
                        else:
                            # allow partial role matches or include if role substring appears in JD text
                            if role.lower() in d.get("text","").lower():
                                filtered.append(d)
                    else:
                        filtered.append(d)
            # Fallback: if no filtered docs, use any retrieved docs (or load specific file text)
            if not filtered:
                filtered = retrieved[:TOP_K]

            # 5) Prepare top_results for display (score not provided by Chroma wrapper, show simple ordering)
            top_results = []
            for i, d in enumerate(filtered[:TOP_K]):
                top_results.append({"score": None, "item": {"filename": d.get("metadata", {}).get("source",""), "text": d.get("text","")}})

            # If still no results, fall back to reading the direct file for company_role
            if not top_results:
                jd_filename_candidates = []
                # try a few filename patterns
                candidates = [
                    f"{company} - {role}.pdf",
                    f"{company}_{role}.pdf",
                    f"{company} - {role} - JD.pdf",
                ]
                for cand in candidates:
                    path = os.path.join(JDS_FOLDER, cand)
                    if os.path.exists(path):
                        jd_filename_candidates.append(path)
                if jd_filename_candidates:
                    # read the first
                    text = extract_text_from_pdf_file(jd_filename_candidates[0])
                    top_results = [{"score": None, "item": {"filename": os.path.basename(jd_filename_candidates[0]), "text": text}}]
                else:
                    st.warning("No JD passages found for the selected company & role.")

            # 6) Show retrieved snippet(s)
            st.markdown("### Retrieved JD snippets (role context)")
            for r in top_results:
                st.write(f"JD file: {r['item'].get('filename')}")
                st.text_area("JD snippet", r['item'].get('text', "")[:1200], height=140)

            # 7) Call LLM for gap analysis (agentic step)
            role_context = "\n\n".join([r['item'].get('text','')[:2000] for r in top_results])
            analysis = call_gap_analysis_llm(cv_text, role_context, program)
            if "error_raw" in analysis:
                st.error("LLM did not return valid JSON. Showing raw output for debugging.")
                st.code(analysis.get("error_raw"))
            else:
                st.session_state["gap_analysis"] = analysis
                st.session_state["last_cv_text"] = cv_text
                st.success("Gap analysis complete.")
                # Compute a simple numeric similarity (optional): compare CV embedding to JD embedding (use first JD)
                try:
                    cv_emb = get_embedding_via_openai(cv_text)
                    jd_emb_sample = get_embedding_via_openai(role_context[:2000])
                    sim = compute_cosine_similarity(cv_emb, jd_emb_sample)
                    st.session_state["score"] = round(sim * 100, 2)
                except Exception:
                    st.session_state["score"] = None

        except Exception as e:
            st.error(f"Analysis failed: {e}")

# --- Generate Plan ---
if plan_btn and st.session_state.get("gap_analysis"):
    months = st.number_input("Months until placement", min_value=1, max_value=24, value=3)
    with st.spinner("Generating weekly plan via LLM..."):
        try:
            # simple planner prompt
            analysis_json = st.session_state["gap_analysis"]
            prompt = f"""
You are an expert learning planner. Return ONLY JSON.
Given analysis: {json.dumps(analysis_json)[:3000]}
months_to_placement: {months}

Task: produce a week-by-week plan to close top prioritized gaps and complete recommended certifications.
Return JSON: {{"analysis_id":"", "months_to_placement":{months}, "feasible":true, "weekly_plan":[], "milestones":[], "est_total_hours":0}}
"""
            resp = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role":"system","content":"You are a helpful planner and return JSON only."},
                          {"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=1000
            )
            raw = ""
            try:
                raw = resp.choices[0].message["content"]
            except Exception:
                raw = str(resp)
            first = raw.find("{"); last = raw.rfind("}")
            if first != -1 and last != -1:
                plan_json = json.loads(raw[first:last+1])
                st.session_state["weekly_plan"] = plan_json
                st.success("Plan generated")
            else:
                st.error("Plan generation failed, LLM returned non-JSON.")
                st.code(raw)
        except Exception as e:
            st.error(f"Plan generation failed: {e}")

# --- Results display (preserve your existing UI) ---
if st.session_state.get("gap_analysis"):
    col1, col2 = st.columns([1, 2])
    with col1:
        # Gauge for Match Score (either computed cosine or LLM-provided match_score if present)
        metric_value = st.session_state.get("score")
        # If LLM returned match_score, prefer that
        try:
            lmmatch = st.session_state["gap_analysis"].get("match_score")
            if lmmatch:
                metric_value = lmmatch
        except Exception:
            pass
        if metric_value is None:
            metric_value = 0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(metric_value),
            title={'text': "Match Score"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green"},
                   'steps': [{'range': [0, 50], 'color': "#ff4d4d"},
                             {'range': [50, 75], 'color': "#ffcc00"},
                             {'range': [75, 100], 'color': "#66cc66"}],}
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Results")
        tab1, tab2, tab3 = st.tabs(["Gap Analysis", "Missing Skills", "Learning Plan"])

        with tab1:
            with st.expander("View Detailed Gap Analysis"):
                st.write(st.session_state["gap_analysis"])

        with tab2:
            st.write("### Missing Skills (from LLM)")
            try:
                st.write(st.session_state["gap_analysis"].get("gaps", []))
            except Exception:
                st.write("No structured gaps available.")

        with tab3:
            if st.session_state.get("weekly_plan"):
                st.write("### Weekly Learning Plan")
                st.write(st.session_state["weekly_plan"])
            else:
                st.info("Click 'Generate Plan' to create a learning plan.")

# End
