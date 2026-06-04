import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix

# Set page layout and title
st.set_page_config(page_title="AutoJudge", page_icon="🏆", layout="centered")

# --- PREMIUM MINIMALIST CSS OVERRIDES ---
st.markdown("""
    <style>
    /* 1. Force absolute font consistency across EVERY element (Shadow DOM bypass) */
    *, *::before, *::after, html, body, [data-testid="stAppViewContainer"], .stApp, p, span, label, input, textarea, button {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Roboto, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* 2. Style the main background to a premium soft gray */
    [data-testid="stAppViewContainer"] {
        background-color: #F1F5F9 !important; /* Soft elegant background */
    }
    
    /* 3. Wrap the main layout inside a beautiful floating application card */
    .main .block-container {
        background-color: #FFFFFF !important;
        border-radius: 20px !important;
        padding: 3rem 3.5rem !important;
        margin-top: 3rem !important;
        margin-bottom: 3rem !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02) !important;
        border: 1px solid #E2E8F0 !important;
        max-width: 680px !important;
    }

    /* Clean Streamlit default header spacing */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* 4. Sleek Typography & Headers */
    .app-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .app-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0F172A; /* Deep dark slate */
        letter-spacing: -0.035em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B; /* Muted steel gray */
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: -0.01em;
    }

    /* 5. Inputs & Labels Formatting */
    label p {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #475569 !important; /* Steel charcoal */
        margin-bottom: 0.5rem !important;
    }
    
    /* Base style for Text Area Containers */
    div[data-baseweb="textarea"] {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #F8FAFC !important; /* Soft pale contrast */
        box-shadow: none !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 4px !important;
    }
    
    /* Focused State */
    div[data-baseweb="textarea"]:focus-within {
        border-color: #0F172A !important; /* Rich black border */
        background-color: #FFFFFF !important; /* Switches to pure white on focus */
        box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.06) !important;
    }
    
    /* Actual textarea element typography styling */
    textarea {
        color: #0F172A !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        font-weight: 400 !important;
    }
    
    /* Placeholders style alignment */
    textarea::placeholder {
        color: #94A3B8 !important;
    }

    /* 6. Pill-shaped Premium Action Button */
    div.stButton > button {
        background-color: #0F172A !important; /* Deep dark slate */
        color: #FFFFFF !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 30px !important; /* Premium rounded pill */
        padding: 0.8rem 2.5rem !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
        margin-top: 1rem;
    }
    
    div.stButton > button:hover {
        background-color: #1E293B !important; /* Slightly lighter on hover */
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.2) !important;
        transform: translateY(-1px);
    }
    
    div.stButton > button:active {
        transform: translateY(1px);
    }

    /* 7. Output Cards Styling */
    .output-card {
        margin-top: 2rem;
        padding: 1.75rem;
        border-radius: 16px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.02);
    }
    
    .card-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1.5rem;
    }
    
    .res-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .res-value-class {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.04em;
    }
    
    .txt-easy { color: #10B981 !important; }
    .txt-medium { color: #F59E0B !important; }
    .txt-hard { color: #EF4444 !important; }
    
    .res-value-score {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.04em;
    }
    .res-score-max {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
    }
    
    /* Summary metadata details bar */
    .summary-info {
        border-top: 1px solid #F1F5F9;
        margin-top: 1.25rem;
        padding-top: 1rem;
        font-size: 0.8rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & PREPROCESSORS ---
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    scaler = joblib.load("feature_scaler.joblib")
    class_model = joblib.load("class_model.joblib")
    reg_model = joblib.load("reg_model.joblib")
    return tfidf, scaler, class_model, reg_model

try:
    tfidf, scaler, class_model, reg_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure joblib files are present in the same directory.")

# --- PREPROCESSING FUNCTIONS ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def count_math_indicators(text):
    indicators = [
        r'\$', r'\\le', r'\\ge', r'==', r'<', r'>', r'\^', r'\\times', 
        r'O\(', r'o\(', r'matrix', r'modulo', r'integer', r'float'
    ]
    count = 0
    for pattern in indicators:
        count += len(re.findall(pattern, text))
    return count

cp_keywords = {
    'graph_terms': r'(graph|vertex|edge|vertices|path|adjacency|shortest path|bfs|dfs)',
    'dp_terms': r'(dynamic programming|dp|memoization|overlapping subproblem|knapsack)',
    'tree_terms': r'(tree|binary tree|bst|node|parent|child|ancestor|lca)',
    'string_terms': r'(substring|string|palindrome|anagram|suffix|prefix|regex)',
    'math_terms': r'(prime|gcd|factorial|combinatorics|probability|modular|geometry|coordinate)'
}

def extract_features(raw_text):
    cleaned = clean_text(raw_text)
    char_count = len(cleaned)
    word_count = len(cleaned.split())
    math_count = count_math_indicators(cleaned)
    
    features = [char_count, word_count, math_count]
    for pattern in cp_keywords.values():
        features.append(len(re.findall(pattern, cleaned.lower())))
        
    numerical_features = np.array(features).reshape(1, -1)
    scaled_numerical = scaler.transform(numerical_features)
    tfidf_features = tfidf.transform([cleaned])
    final_features = hstack([tfidf_features, csr_matrix(scaled_numerical)])
    return final_features

# --- UI HEADER ---
st.markdown("""
    <div class="app-header">
        <div class="app-title">AutoJudge</div>
        <div class="app-subtitle">Predict coding problem difficulty class and score using NLP</div>
    </div>
""", unsafe_allow_html=True)

# Form Fields
problem_desc = st.text_area("Problem Description", placeholder="Paste the core problem statement here...", height=150)
input_desc = st.text_area("Input Format", placeholder="Describe input formatting and parameter limits...", height=100)
output_desc = st.text_area("Output Format", placeholder="Describe expected output structure...", height=100)

# Evaluate Trigger
if st.button("Evaluate"):
    if not problem_desc.strip():
        st.warning("Please provide a problem description to perform an evaluation.")
    else:
        combined_text = f"{problem_desc} {input_desc} {output_desc}"
        processed_features = extract_features(combined_text)
        
        predicted_class = class_model.predict(processed_features)[0].lower()
        predicted_score = reg_model.predict(processed_features)[0]
        predicted_score = max(1.1, min(9.7, predicted_score))
        
        cls_color = f"txt-{predicted_class}"
        
        # Display clean Apple-style output card
        st.markdown(f"""
            <div class="output-container">
                <div class="output-card">
                    <div class="card-row">
                        <div>
                            <div class="res-label">Predicted Class</div>
                            <div class="res-value-class {cls_color}">{predicted_class.upper()}</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="res-label">Difficulty Score</div>
                            <div class="res-value-score">
                                {predicted_score:.2f} <span class="res-score-max">/ 9.70</span>
                            </div>
                        </div>
                    </div>
                    <div class="summary-info">
                        <span>📊</span>
                        <span>Evaluation complete. Analyzed {len(combined_text.split())} words and {count_math_indicators(combined_text)} mathematical indicators.</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)