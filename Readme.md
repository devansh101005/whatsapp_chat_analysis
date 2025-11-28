#  WhatsApp Chat Analysis & Behavioral Analytics

A **Streamlit-based NLP & Machine Learning application** for advanced analysis of WhatsApp chat exports.  
This project goes beyond basic message counting and introduces **sentiment analysis, topic modeling, toxicity detection, and social network graph analysis** for group and personal chats.

---

##  Features

### Chat Statistics
- Total messages, words, media, and links  
- User-wise & overall participation  
- Daily and monthly timelines  
- Weekly activity heatmaps  

### Text & Emoji Analysis
- WordCloud with Hinglish stopword removal  
- Most common words  
- Emoji frequency and distribution  

###  Sentiment Analysis (Multiple Models)
- Logistic Regression (TF-IDF based)
- Support Vector Machine (SVM)
- Multilingual BERT  
Supports English, Hindi, and Hinglish chats.

###  Topic Modeling (LDA)
- Latent Dirichlet Allocation (LDA)
- Automatic discovery of discussion topics
- Adjustable number of topics
- Noise removal using custom blocked words

###  Toxicity & Abuse Detection (Innovation)
- Transformer-based Toxic-BERT model
- Detects:
  - Toxic
  - Insult
  - Obscene
  - Threat
  - Identity hate
- Highlights most toxic messages
- Category-wise toxicity distribution

### Social Network Graph Analysis (Innovation)
- Directed conversation graph
- Nodes represent users
- Edges represent interaction flow
- Static graph using NetworkX
- Interactive graph using PyVis
- Identifies influential and bridging users

---

## 🛠 Tech Stack

- Frontend: Streamlit  
- NLP: NLTK, Gensim, WordCloud  
- Machine Learning: Scikit-learn (LogReg, SVM)  
- Deep Learning: Transformers, PyTorch  
- Visualization: Matplotlib, Seaborn, PyVis  
- Graph Analysis: NetworkX  

---

##  Quick Start

### 1️Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
 ```
## Install Dependencies
pip install streamlit pandas numpy scikit-learn joblib nltk transformers torch gensim pyvis emoji deep-translator matplotlib seaborn wordcloud urlextract networkx
python -m nltk.downloader stopwords

## Run the Application

cd whatsapp-chat-analysis
streamlit run app.py

## Project Structure 

whatsapp-chat-analysis/
│
├── app.py                     # Streamlit UI & routing
├── preprocessor.py            # Robust WhatsApp chat parser
├── helper.py                  # Analysis & ML utilities
│
├── models/
│   ├── sentiment_model.pkl
│   ├── sentiment_vectorizer.pkl
│   ├── svm_sentiment_model.pkl
│   └── svm_vectorizer.pkl
│
├── stop_hinglish.txt           # Hinglish stopwords
├── chat_network.html           # Interactive graph output
│
├── sentiment_model_train.py    # Logistic Regression training
├── train_svm.py                # SVM training
├── test_sentiment.py           # Sentiment tests
│
└── README.md


## Model Training 
python sentiment_model_train.py
python train_svm.py


## Accuracy Of Logistical Regression 
Training Accuracy: 77.11%
Test Accuracy: 76.78 %

## Accuracy Of SVM 
Training Accuracy: 80.95 %
Test Accuracy: 73.79 %

BERT stands for Bidirectional Encoder Representations from Transformers.
TF-IDF (Term Frequency–Inverse Document Frequency).

LDA (Latent Dirichlet Allocation)?

Unlike ML models, BERT captures bidirectional context using self-attention.”
