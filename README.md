# AutoJudge: Algorithmic Complexity Evaluation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

AutoJudge is an intelligent machine learning system designed to automatically evaluate the difficulty of programming tasks. Using only raw, unstructured text descriptions from coding problems (such as those hosted on Codeforces, Kattis, or CodeChef), AutoJudge predicts both the qualitative difficulty class (Easy, Medium, Hard) and a fine-grained numerical complexity score (1.1 to 9.7).

---

## System Architecture & Methodology

AutoJudge employs a hybrid feature extraction pipeline designed to capture both semantic vocabulary meaning and domain-specific structural indicators:

```text
  [ Raw Description Text ]
              │
              ▼
    [ Text Preprocessing ] ────► Removes HTML tags & normalizes spacing
              │
              ├─────────────────────────────────────────┐
              ▼                                         ▼
   [ Lexical Vectors (TF-IDF) ]            [ 8-Dimensional Structural Features ]
   - Sublinear scaling                     - Character & Word count
   - Top 3,000 unigrams & bigrams          - CP Topic occurrences (Graph, DP, Math, etc.)
              │                            - LaTeX & Mathematical token frequency
              │                                         │
              │                                         ▼
              │                               [ StandardScaler ]
              │                                         │
              └───────────────────┬─────────────────────┘
                                  ▼
                        [ Horizontal Stacking ] ──► Matrix Shape: (N, 3008)
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
  [ Classifier (Logistic Regression) ]    [ Regressor (Random Forest) ]
  - Predicts Class (Easy/Medium/Hard)     - Predicts Score (1.10 - 9.70)
```

---

## Dataset & Model Performances

The underlying models were trained on **4,112 verified competitive programming problems** parsed with an 80/20 train-test split.

### 1. Classification Model (Logistic Regression)
- **Overall Accuracy:** `46.90%`
- **Balanced Class Weights:** Due to class imbalances (only 18% of tasks are Easy, while 47% are Hard), the model was optimized using `class_weight='balanced'` to prevent bias toward the majority class. This ensures high structural representation across all three targets:
  - **Easy Class F1-Score:** `0.44`
  - **Medium Class F1-Score:** `0.36`
  - **Hard Class F1-Score:** `0.56`

### 2. Regression Model (Random Forest Regressor)
- **Mean Absolute Error (MAE):** `1.6793` (On a scale of 1.10 to 9.70, predictions deviate by an average of only ~1.68 points)
- **Root Mean Squared Error (RMSE):** `2.0362`
- **R-squared ($R^2$):** `0.1363`

---

## Project Structure

The version control structure of this project is organized as follows:

```text
autojudge/
|-- data/
|   +-- problems_data.jsonl        --> Raw problem dataset
|-- exploratory_analysis.ipynb     --> EDA & model training notebook
|-- app.py                         --> Styled Streamlit web application
|-- cleaned_problems_data.csv      --> Preprocessed text data cache
|-- tfidf_vectorizer.joblib        --> Saved TF-IDF vectorizer
|-- feature_scaler.joblib          --> Saved structural feature scaler
|-- class_model.joblib             --> Trained Logistic Regression model
|-- reg_model.joblib               --> Trained Random Forest Regressor
|-- requirements.txt               --> Standardized Python dependencies
|-- README.md                      --> Markdown repository documentation
```

---

## Local Installation & Setup

Ensure you have Python 3.10+ installed on your local machine.

### 1. Clone the repository and navigate inside:
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/autojudge.git
cd autojudge
```

### 2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit web application:
```bash
streamlit run app.py
```
This launches a local web server. Open `http://localhost:8501` in your browser to evaluate problem complexity.
