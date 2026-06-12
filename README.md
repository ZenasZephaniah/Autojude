### Project Structure
```text
autojudge/
├── data/
│   └── problems_data.jsonl       # Original raw dataset
├── venv/                         # Python virtual environment (ignored by Git)
├── exploratory_analysis.ipynb    # Jupyter Notebook for EDA, Feature Engineering & Training
├── app.py                        # Clean, styled Streamlit UI application
├── cleaned_problems_data.csv     # Preprocessed data artifact
├── tfidf_vectorizer.joblib       # Saved TF-IDF Vectorizer
├── feature_scaler.joblib         # Saved StandardScaler for structural features
├── class_model.joblib            # Trained Logistic Regression classifier
├── reg_model.joblib              # Trained Random Forest Regressor
├── requirements.txt              # Standardized python dependencies
└── README.md                     # Project documentation
```

### Local Installation & Setup

Ensure you have Python 3.10+ installed on your system.

### 1. Clone the repository and navigate inside:
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/autojudge.git
cd autojudge
```

### 2. Set up and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install all project dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit web application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser to test the interface.
