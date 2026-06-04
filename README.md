# AutoJudge: Programmatic Difficulty & Complexity Predictor

AutoJudge is an end-to-end machine learning system designed to automatically analyze competitive programming problem descriptions (e.g., from platforms like Codeforces, Kattis, or CodeChef) and predict their difficulty class and numerical complexity score.

By processing only the textual context of the problem—the description, input constraints, and output requirements—the model extracts structural complexity markers and semantics to generate its predictions.

## Features
- **Dual-Model Inference:** Performs classification (predicting Easy/Medium/Hard) and regression (predicting a numerical difficulty score from 1.1 to 9.7) simultaneously.
- **Hybrid Feature Engineering:** Combines high-dimensional TF-IDF lexical vectors with 8 domain-specific structural features extracted directly from competitive programming syntax.
- **Chic, Minimalist UI:** Built using a custom CSS override on Streamlit, heavily inspired by modern design interfaces (such as Apple and Spotify's design systems).

---

## Tech Stack & Architecture
- **Language:** Python 3.12
- **Core ML Libraries:** Scikit-Learn, Pandas, NumPy, SciPy, Joblib
- **Web UI Framework:** Streamlit (with custom CSS injection)
- **Model Storage:** Serialization using Joblib binary files

---

## Project Structure
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

---

## Dataset & Preprocessing Pipeline
The pipeline is trained on **4,112 verified programming tasks** possessing the following characteristics:
- **Class Distribution:** Hard (1,941 samples), Medium (1,405 samples), Easy (766 samples). Due to this class imbalance, models were optimized to balance class weights rather than targeting raw accuracy alone.
- **Score Scale:** Range from 1.10 (easiest) to 9.70 (hardest) with a mean value of 5.11.

### Preprocessing & Feature Extraction
To capture both structural syntax and semantic meaning, the system parses incoming text through two distinct feature extractions:

1. **Text Normalization:** Eliminates heavy HTML layout tags and normalizes multiple line breaks while preserving mathematical expressions.
2. **TF-IDF Representation:** Converts the unified text string into a sparse vector of 3,000 top n-grams (unigrams and bigrams), capturing semantic keywords.
3. **Engineered Structural Features (8 Dimensions):**
   - **Text Lengths:** Character and word counts (correlates with problem reading complexity).
   - **Math Indicator Count:** Frequency of mathematical expressions, inequality markers (`\le`, `\ge`, `==`), LaTeX symbols, and complexity notation (`O(N)`).
   - **Competitive Programming (CP) Topic Frequency:** Keyword counts matching core data structures and algorithms (Graph theory, Dynamic Programming, Trees, String processing, and Advanced Math).

---

## Model Performance & Evaluation

The training data was split using an 80/20 train-test ratio. The models selected for production deployment yielded the following test metrics:

### 1. Classification (Logistic Regression with Balanced Class Weights)
- **Overall Accuracy:** 46.90%
- **Evaluation Strategy:** Due to class imbalances, we utilized balanced class weighting. This prevented the model from ignoring the minority class (Easy), ensuring balanced class-wise F1-scores:
  - **Easy:** F1-Score of 0.44
  - **Medium:** F1-Score of 0.36
  - **Hard:** F1-Score of 0.56

### 2. Regression (Random Forest Regressor)
- **Mean Absolute Error (MAE):** 1.6793 (On a scale of 1.1 to 9.7, predictions deviate by an average of ~1.68 points)
- **Root Mean Squared Error (RMSE):** 2.0362
- **R-squared Score ($R^2$):** 0.1363

---

## Local Installation & Setup

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
```
