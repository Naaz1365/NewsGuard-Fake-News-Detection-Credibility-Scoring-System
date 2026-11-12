
# NewsGuard — Model Card

## 1. Model Overview

- **Project Name:** NewsGuard — Fake News Detection System
- **Model Type:** Extra Trees Classifier
- **Task:** Binary Text Classification
- **Framework:** Scikit-learn
- **Dataset:** LIAR Dataset
- **Primary Objective:** Classify news statements as REAL or FAKE.

NewsGuard is an educational machine learning prototype that analyzes text patterns and estimates whether a news statement belongs to the REAL or FAKE class.

## 2. Intended Use

- Educational and academic projects.
- Machine learning and NLP demonstrations.
- Research and experimentation with text classification.
- Demonstration of credibility scoring.

### Out-of-Scope Use

- Making legal, financial, medical, or political decisions.
- Treating predictions as verified facts.
- Replacing professional fact-checking organizations.
- Making automated decisions about individuals or organizations.

## 3. Dataset Information

- **Dataset:** LIAR
- **Original Labels:** Six truthfulness categories.
- **Final Classes:** REAL and FAKE.
- **Training Samples:** 6,434
- **Validation Samples:** 804
- **Test Samples:** 805

### Label Mapping

- REAL: true, mostly-true
- FAKE: false, pants-fire
- barely-true and half-true categories were excluded.

## 4. Data Preprocessing

The preprocessing pipeline includes:

- Lowercase conversion.
- URL and email removal.
- HTML tag removal.
- Special character removal.
- Tokenization using NLTK.
- English stopword removal.
- Word lemmatization.

## 5. Feature Engineering

The model uses the following features:

1. TF-IDF — captures word importance.
2. Word2Vec — captures semantic word representations.
3. Flesch Reading Ease — measures text readability.
4. VADER Sentiment — measures sentiment polarity.

Total feature dimensions: 9,983.

## 6. Model Selection

The following classifiers were evaluated:

- Logistic Regression
- Linear SVM
- Random Forest
- Extra Trees
- SGD Classifier

Extra Trees was selected based on validation FAKE-class F1 performance.

## 7. Evaluation Metrics

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

## 8. Explainability

Feature importance is generated using the Extra Trees model's `feature_importances_` attribute.

The project saves:
- Feature importance chart.
- Feature importance CSV.

SHAP-based explanations are not currently available in this environment.

## 9. Limitations

- The model is trained on the LIAR dataset and may not generalize to all news sources.
- Performance depends on text quality and training data.
- The model can produce false positives and false negatives.
- Readability and sentiment are supporting features, not indicators of factual truth.
- The credibility score represents the model-estimated REAL probability, not verified factual accuracy.
- Dataset bias and domain differences may affect predictions.

## 10. Ethical Considerations

NewsGuard should not be used as the sole authority for determining whether a news statement is true or false.

Users should verify important claims through reliable sources and professional fact-checking organizations.

## 11. Disclaimer

NewsGuard is an educational machine learning prototype. Its predictions and credibility scores are estimates based on patterns learned from the LIAR dataset. They do not establish factual accuracy.

## 12. Project Status

- Dataset preprocessing: Completed
- Feature engineering: Completed
- Model training: Completed
- Model evaluation: Completed
- Prediction pipeline: Completed
- Streamlit dashboard: Completed
- Feature importance: Completed
- MLflow tracking: Completed
- Automated tests: 6 passed

**Project:** NewsGuard — Fake News Detection System  
**Model:** Extra Trees Classifier  
**Version:** 1.0
