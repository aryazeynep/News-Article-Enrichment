# 📰 News Article Enrichment with BERT and Google Gemini API

This project demonstrates an end-to-end solution using transformer-based models to solve a real-world scenario involving news article classification and enhancement.

---

## 📌 Problem Statement

You are managing a news website with a collection of articles from various sources. However, these articles:
- Do not have category labels.
- Have titles embedded inside the article body, not clearly separated.

Your goal is to enrich each article using AI technologies by:
1. Classifying the category of each article using a fine-tuned BERT model.
2. Extracting and professionally rewriting the article and its headline using Google Gemini API.
3. Returning the enriched article in a structured format: `{ category, title, article }`.

---

## 📂 Dataset

We use a subset of the [AG News dataset](https://huggingface.co/datasets/ag_news) available on Hugging Face:

- **Training Set**: 30,000 samples (downsampled from the original 120,000 for efficiency)
- **Test Set**: 7,600 samples
- **Number of Classes**: 4
  - World
  - Sports
  - Business
  - Sci/Tech

---

## 🛠 Tools & Environment

- **Model**: BERT Base Uncased (`bert-base-uncased`)
- **API**: Google Gemini API (via Google AI Studio)
- **Notebook**: Developed using Jupyter Notebook (Google Colab T4 GPU-compatible)
- **Libraries**: `transformers`, `datasets`, `scikit-learn`, `torch`, `matplotlib`, `pandas`, `openai` (or `google.generativeai`)

---

## 🔍 Part 1: Article Classification

- Conducted exploratory data analysis (EDA) to understand class distribution and article length.
- Fine-tuned a `BertForSequenceClassification` model on the training set.
- Evaluated model performance using the weighted **F1 score** on the test set.
- Performed hyperparameter tuning to improve results.
- **Achieved F1 Score**: > 0.90
- Saved the final model for later use.
- Example predictions on first 5 test samples were included.

---

## ✍️ Part 2: Title and Well-Written Article Generation

- Integrated **Google Gemini API** to process raw articles.
- Used prompting techniques to ensure:
  - Clear extraction of **title**
  - Polished, professional **rewriting of the article**
- Extracted title and article separately using regular expressions or string parsing.

---

## 🔗 Part 3: Combined Enrichment Pipeline

Created a complete enrichment pipeline:
1. Load an article
2. Predict its category using the saved BERT model
3. Generate title and rewritten article using Gemini
4. Output a dictionary:
```json
{
  "category": "Business",
  "title": "US Stocks Rally as Fed Signals Rate Pause",
  "article": "The US stock market saw a sharp increase today as Federal Reserve officials..."
}
