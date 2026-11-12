
# 🛡️ NewsGuard — Fake News Detection System

An AI-powered Fake News Detection System using Machine Learning, Natural Language Processing (NLP), and Explainable AI techniques.

## 📌 Project Overview

NewsGuard is an educational machine learning project designed to classify news statements into two categories:

- REAL
- FAKE

The system analyzes text using multiple NLP and machine learning features and generates a model-estimated credibility score.

**Note:** The credibility score is an estimate, not proof of factual accuracy.

## ✨ Key Features

- Fake and real news classification
- TF-IDF text feature extraction
- Word2Vec semantic embeddings
- Readability analysis using Flesch Reading Ease
- Sentiment analysis using VADER
- Extra Trees classification model
- Credibility score estimation
- Feature importance visualization
- MLflow experiment tracking
- Interactive Streamlit dashboard
- Automated testing with Pytest

## 🧠 Technologies Used

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn |
| NLP | NLTK, Gensim |
| Feature Extraction | TF-IDF, Word2Vec |
| Sentiment Analysis | VADER |
| Readability | Textstat |
| Visualization | Matplotlib, Streamlit |
| Experiment Tracking | MLflow |
| Testing | Pytest |

## 📂 Project Structure

```text
NewsGuard/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── features/
│
├── models/
│
├── outputs/
│   └── plots/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── predict.py
│   ├── explainability.py
│   └── mlflow_tracking.py
│
├── tests/
│   └── test_newsguard.py
│
├── dashboard.py
├── MODEL_CARD.md
├── README.md
└── requirements.txt
```

## 📊 Dataset

**Dataset:** LIAR Dataset

The original dataset contains six truthfulness categories. For this project, labels were mapped into two classes:

- REAL: true, mostly-true
- FAKE: false, pants-fire

Barely-true and half-true samples were excluded.

| Dataset Split | Samples |
|---|---:|
| Training | 6,434 |
| Validation | 804 |
| Testing | 805 |

## ⚙️ Feature Engineering

The system combines four types of features:

1. **TF-IDF:** Represents the importance of words and phrases.
2. **Word2Vec:** Captures semantic relationships between words.
3. **Readability:** Uses Flesch Reading Ease.
4. **Sentiment:** Uses VADER compound sentiment score.

Total feature dimensions: **9,983**.

## 🤖 Model Training

Five classification algorithms were evaluated:

- Logistic Regression
- Linear SVM
- Random Forest
- Extra Trees
- SGD Classifier

The Extra Trees Classifier was selected based on validation performance.

## 📈 Model Performance

### Held-Out Test Results

| Metric | Score |
|---|---:|
| Accuracy | 66.09% |
| FAKE Precision | 60.15% |
| FAKE Recall | 67.80% |
| FAKE F1 Score | 63.75% |

### Classification Report

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| FAKE | 0.60 | 0.68 | 0.64 |
| REAL | 0.72 | 0.65 | 0.68 |

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Naaz1365/NewsGuard.git
cd NewsGuard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
python -m streamlit run dashboard.py
```

Open:

```text
http://localhost:8501
```

## 🔍 Run Prediction from Terminal

```bash
python src/predict.py
```

Enter a news statement when prompted.

## 🧪 Run Automated Tests

```bash
python -m pytest tests/ -v
```

Current test status: **6 tests passed**.

## 📊 Explainability

Feature importance is generated using the Extra Trees model.

Generated files:

- `outputs/feature_importance.csv`
- `outputs/plots/feature_importance.png`

## 📉 MLflow Tracking

MLflow is used to record:

- Model parameters
- Evaluation metrics
- Experiment runs
- Feature importance artifacts

## ⚠️ Limitations and Disclaimer

- This is an educational machine learning prototype.
- Predictions are based on patterns learned from the LIAR dataset.
- The model may produce incorrect classifications.
- Readability and sentiment are supporting features, not indicators of truth.
- Credibility scores are model-estimated probabilities, not verified facts.
- Always verify important news through reliable sources.

## 👩‍💻 Author

**Nazneen Fatema Asad Baig**

- GitHub: [Naaz1365](https://github.com/Naaz1365)
- LinkedIn: [Nazneen Fatema Baig](https://www.linkedin.com/in/nazneenfatema-baig-2062a9380/)

---

**NewsGuard — Fake News Detection System**

Machine Learning • NLP • Explainable AI
