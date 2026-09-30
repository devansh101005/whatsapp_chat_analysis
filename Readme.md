# WhatsApp Chat Analysis & Behavioral Analytics

A **Streamlit-based NLP & Machine Learning application** for analyzing WhatsApp chat exports.
It goes beyond basic message counting and adds **sentiment analysis, topic modeling, toxicity
detection, and social network graph analysis** for group and personal chats.

---

## Features

### Chat Statistics
- Total messages, words, media, and links
- User-wise & overall participation
- Daily and monthly timelines
- Weekly activity heatmaps
- Busiest users, days, and months

### Text & Emoji Analysis
- WordCloud with Hinglish stopword removal
- Most common words
- Emoji frequency and distribution

### Sentiment Analysis (Multiple Models)
- Logistic Regression (TF-IDF based)
- Support Vector Machine (SVM)
- Multilingual BERT
- A small Hinglish slang lexicon to correct strong slang words
- Supports English, Hindi, and Hinglish chats (auto-translation to English)

### Topic Modeling (LDA)
- Latent Dirichlet Allocation to discover discussion themes
- Adjustable number of topics
- Noise removal using custom blocked words

### Toxicity & Abuse Detection
- Transformer-based Toxic-BERT model
- Detects toxic, insult, obscene, threat, and identity-hate content
- Highlights the most toxic messages

### Social Network Graph Analysis
- Directed conversation graph (who replies to whom)
- Static graph using NetworkX
- Interactive graph using PyVis

---

## Tech Stack

- **Frontend:** Streamlit
- **NLP:** NLTK, Gensim, WordCloud
- **Machine Learning:** scikit-learn (Logistic Regression, SVM)
- **Deep Learning:** Transformers, PyTorch
- **Visualization:** Matplotlib, Seaborn, PyVis
- **Graph Analysis:** NetworkX

---

## Quick Start

### 1. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate     # Windows (Git Bash): source .venv/Scripts/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

### 3. Run the application
```bash
streamlit run app.py
```

Then upload a WhatsApp chat `.txt` file (exported via **Export Chat → Without Media**).

---

## Sentiment Models

The Logistic Regression and SVM models are trained on the
[Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).

The trained `.pkl` files are not committed to the repo. To create them, download the
Sentiment140 CSV into the project root and run:

```bash
python sentiment_model_train.py   # creates sentiment_model.pkl + sentiment_vectorizer.pkl
python train_svm.py               # creates svm_sentiment_model.pkl + svm_vectorizer.pkl
```

The app loads these models lazily (only when the Sentiment page is opened), so it still
starts even if the model files are missing. BERT and Toxic-BERT are downloaded
automatically from Hugging Face the first time they are used.

### Reported Accuracy (Sentiment140)
| Model               | Train Accuracy | Test Accuracy |
|---------------------|----------------|---------------|
| Logistic Regression | 77.11%         | 76.78%        |
| SVM (LinearSVC)     | 80.95%         | 73.79%        |

You can regenerate evaluation plots (confusion matrix, ROC, feature importance,
model comparison) with:
```bash
python model_evaluation.py
```

---

## Running the Tests
```bash
pytest
```
The tests cover the chat parser and the core helper functions. Tests that need the
heavy ML libraries are skipped automatically if those libraries are not installed.

---

## Running with Docker
```bash
docker build -t whatsapp-chat-analysis .
docker run -p 8501:8501 whatsapp-chat-analysis
```
Then open http://localhost:8501 in your browser.

The app can also be deployed for free on **Streamlit Community Cloud** or
**Hugging Face Spaces** by pointing it at this repository and using `app.py` as the
entry point.

---

## Project Structure
```
whatsapp-chat-analysis/
│
├── app.py                      # Streamlit UI & page routing
├── preprocessor.py             # WhatsApp chat parser (Android + iPhone formats)
├── helper.py                   # Analysis & ML utilities
│
├── sentiment_model_train.py    # Logistic Regression training
├── train_svm.py                # SVM training
├── model_evaluation.py         # Evaluation plots (confusion matrix, ROC, etc.)
├── test_sentiment.py           # Manual sentiment smoke test
│
├── tests/                      # Automated pytest tests
│   ├── test_preprocessor.py
│   └── test_helper.py
│
├── stop_hinglish.txt           # Hinglish stopwords
├── requirements.txt            # Pinned dependencies
└── Readme.md
```

---

## Methodology & Limitations

This is a learning/portfolio project, so it is worth being clear about the trade-offs:

- **Domain shift:** The Logistic Regression and SVM models are trained on English tweets
  (Sentiment140) but applied to Hinglish chat messages. Accuracy on real chats is lower
  than the reported test accuracy because the data is different in style and language.
- **Lexicon override:** A small hand-built Hinglish slang lexicon overrides the ML
  prediction for strongly positive/negative slang words. This improves results on Hinglish
  but means the final sentiment is part model, part rules.
- **Translation dependency:** Non-English messages are translated to English with
  `deep-translator`, which needs an internet connection and can be slow on large chats.
- **Date formats:** The parser supports common Android and iPhone export formats, but very
  unusual locale formats may not parse and those messages are dropped.
- **BERT cost:** The transformer models (BERT, Toxic-BERT) run on CPU by default and can be
  slow on long chats.

---

## Glossary
- **TF-IDF** – Term Frequency–Inverse Document Frequency
- **BERT** – Bidirectional Encoder Representations from Transformers
- **LDA** – Latent Dirichlet Allocation
