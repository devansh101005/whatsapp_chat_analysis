
```markdown
// filepath: c:\Users\hp\PycharmProjects\whatsapp-chat-analysis\DEVELOPER_GUIDE_RETRAIN.md
...existing code...
# Condensed Developer Guide — Model Retraining

Workspace: C:\Users\hp\PycharmProjects\whatsapp-chat-analysis (Windows)

Prerequisites
- Python 3.8+ in a virtual environment.
- Install dependencies:
```powershell
pip install -r C:\Users\hp\PycharmProjects\whatsapp-chat-analysis\requirements.txt
# fallback:
pip install pandas numpy scikit-learn joblib nltk transformers torch gensim pyvis streamlit emoji deep-translator
python -m nltk.downloader stopwords
```

1) Locate training data
- Default large dataset: [training.1600000.processed.noemoticon.csv](http://_vscodecontentref_/2) in repo root.
- Custom CSV must include text and label columns (adjust training script if names differ).

2) Retrain SVM (train_svm.py)
- Open [train_svm.py](http://_vscodecontentref_/3) and confirm TRAIN_CSV path and sampling/hyperparams.
- Run:
```powershell
cd C:\Users\hp\PycharmProjects\whatsapp-chat-analysis
python train_svm.py
```
- Output: typically [svm_sentiment_model.pkl](http://_vscodecontentref_/4) and [svm_vectorizer.pkl](http://_vscodecontentref_/5) (verify script).

3) Retrain Logistic/ML model (sentiment_model_train.py)
- Edit TRAIN_CSV, sample size, hyperparams as needed.
- Run:
```powershell
python sentiment_model_train.py
```
- Output: [sentiment_model.pkl](http://_vscodecontentref_/6) and [sentiment_vectorizer.pkl](http://_vscodecontentref_/7) (or names used in script).

4) Standardize artifact storage
- Create `models\` directory in project root.
- Recommended filenames:
  - models\sentiment_model.pkl
  - models\sentiment_vectorizer.pkl
  - models\svm_sentiment_model.pkl
  - models\svm_vectorizer.pkl
- Example save snippet to add to training scripts:
```python
# language: python
import joblib
# ...existing code...
joblib.dump(trained_model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/sentiment_vectorizer.pkl")
```

5) Minimal validation
- Ensure training scripts evaluate on a hold-out set or cross-validation.
- Use [test_sentiment.py](http://_vscodecontentref_/8) for a quick smoke test:
```powershell
python test_sentiment.py
```

6) Update app model paths
- Confirm [helper.py](http://_vscodecontentref_/9) loads models from `models/` or update paths accordingly.
- Example change:
```python
# filepath: c:\Users\hp\PycharmProjects\whatsapp-chat-analysis\helper.py
# ...existing code...
SVM_MODEL_PATH = "models/svm_sentiment_model.pkl"
SVM_VECT_PATH = "models/svm_vectorizer.pkl"
ML_MODEL_PATH = "models/sentiment_model.pkl"
ML_VECT_PATH = "models/sentiment_vectorizer.pkl"
# ...existing code...
```

7) BERT / Transformer fine-tuning (brief)
- Prepare dataset, tokenize with a Hugging Face tokenizer, use Trainer or custom loop.
- Save model & tokenizer:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# after training:
model.save_pretrained("models/bert_sentiment")
tokenizer.save_pretrained("models/bert_sentiment")
```
- Load in [helper.py](http://_vscodecontentref_/10) via transformers pipeline or AutoModelForSequenceClassification.from_pretrained.

8) Smoke test full app
```powershell
streamlit run C:\Users\hp\PycharmProjects\whatsapp-chat-analysis\app.py
# Upload a sample chat and verify sentiment pages work and predictions display.
```

9) Troubleshooting
- FileNotFoundError on model load: check filenames and [helper.py](http://_vscodecontentref_/11) paths.
- Version mismatches: train and serve with same scikit-learn/joblib versions.
- Memory/timeout issues for transformers: use smaller batch sizes or GPU.

Optional automation
- Add a PowerShell script or Makefile to run training and save artifacts to models\ automatically.

If you want, I can generate the exact save/load code to insert into the training scripts or create a PowerShell script that automates retraining and artifact placement.