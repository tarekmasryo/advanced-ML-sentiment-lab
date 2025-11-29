# 🧪 Advanced ML Sentiment Lab


[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)<br>
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-orange.svg)](LICENSE)<br>
[![Made by Tarek Masryo](https://img.shields.io/badge/Made%20by-Tarek%20Masryo-blue)](https://github.com/tarekmasryo)

---

## 📌 Overview

**Advanced ML Sentiment Lab** is an interactive **Streamlit + Plotly** dashboard for **binary sentiment analysis**.

Give it any CSV with a **text column** and a **binary label** and it will:

- Clean and tokenize the text, then build **TF–IDF word + optional char features**
- Train several classical models (Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes)
- Evaluate them with **ROC / PR AUC, F1, Accuracy, Precision, Recall**
- Let you **tune the decision threshold** with custom FP/FN business costs
- Explore **error analysis** and run **live predictions** on arbitrary text

It works great with the classic **IMDB 50K Reviews** dataset  
([Kaggle link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)),  
but is generic enough for product reviews, tickets, surveys, and comments.

---

## 📊 Dashboard Preview

### EDA & KPIs  
![EDA](assets/eda-hero.png)

### Train & Validation  
![Train & Validation](assets/train-validation.png)

### Error Analysis  
![Error Analysis](assets/error-analysis.png)

### Deploy & Interactive Prediction  
![Deploy](assets/deploy.png)

---

## 🔑 Main Features

- **Flexible input**
  - Upload your own CSV or auto-load common filenames (`IMDB Dataset.csv`, `imdb.csv`, `reviews.csv`, `data.csv`, `comments.csv`)
  - Map text/label columns and choose which values are *positive* vs *negative*

- **Multi-model training**
  - TF–IDF word n-grams (1–3) + optional char n-grams (3–6)
  - Configurable max features and validation split
  - Logistic Regression, Random Forest, Gradient Boosting, Multinomial Naive Bayes

- **Evaluation & comparison**
  - Single stratified train/validation split on a capped subset
  - Metrics per model: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Model comparison cards + ROC & PR curves + confusion matrices

- **Threshold & business cost**
  - Move the decision threshold and see how metrics change
  - Attach **FP / FN costs** and visualise F1 vs threshold and cost vs threshold

- **Error analysis**
  - Browse **false positives / false negatives**
  - Sort by most confident errors or least confident predictions

- **Deploy tab**
  - Pick the **best-by-F1** model or any trained model
  - Predict sentiment for arbitrary text with a clean UI + confidence bar
  - Reuses saved artifacts under `models_sentiment_lab/` (models, vectorisers, metadata)

---

## 🚀 Run Locally

Clone the repo and install dependencies:

```bash
git clone https://github.com/tarekmasryo/advanced-ml-sentiment-lab.git
cd advanced-ml-sentiment-lab

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Place your CSV next to `app.py` or upload it from the sidebar, then map the columns and start experimenting.

---

If you use this dashboard, please credit:

> Advanced ML Sentiment Lab by **Tarek Masryo**.  
> Code licensed under Apache 2.0.
