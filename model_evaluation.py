import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------
# 1. LOAD TRAINING DATA (Sentiment140)
# -------------------------

def load_sentiment140(csv_path, sample_size=50000):
    """
    Loads Sentiment140 dataset and returns a sampled DataFrame
    with columns: ['sentiment', 'text'] where sentiment is 0/1.
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path, encoding="latin-1", header=None)
    # Columns: 0=target, 5=text
    df = df[[0, 5]]
    df.columns = ["sentiment", "text"]

    # Map 0 -> 0 (negative), 4 -> 1 (positive)
    df["sentiment"] = df["sentiment"].replace({4: 1})

    # Optional: sample for speed
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    print(f"Dataset loaded with {len(df)} rows.")
    return df


# -------------------------
# 2. LOAD MODELS & VECTORIZERS
# -------------------------

def load_models():
    print("Loading models and vectorizers...")
    logreg_model = pickle.load(open("sentiment_model.pkl", "rb"))
    logreg_vectorizer = pickle.load(open("sentiment_vectorizer.pkl", "rb"))

    svm_model = pickle.load(open("svm_sentiment_model.pkl", "rb"))
    svm_vectorizer = pickle.load(open("svm_vectorizer.pkl", "rb"))

    print("Models loaded successfully.")
    return logreg_model, logreg_vectorizer, svm_model, svm_vectorizer


# -------------------------
# 3. EVALUATE MODEL + GENERATE CONFUSION MATRIX, ROC, REPORT
# -------------------------

def evaluate_and_plot(df, model, vectorizer, model_name_prefix="logreg"):
    """
    df: DataFrame with columns sentiment (0/1), text
    model: trained classifier
    vectorizer: TF-IDF vectorizer
    model_name_prefix: "logreg" or "svm" (used in file names)
    """
    print(f"\nEvaluating {model_name_prefix.upper()}...")

    X_text = df["text"]
    y_true = df["sentiment"].values

    # Transform text to TF-IDF
    X = vectorizer.transform(X_text)

    # Predict labels
    y_pred = model.predict(X)

    # ---- Metrics ----
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"{model_name_prefix.upper()} Accuracy:  {acc:.4f}")
    print(f"{model_name_prefix.upper()} Precision: {prec:.4f}")
    print(f"{model_name_prefix.upper()} Recall:    {rec:.4f}")
    print(f"{model_name_prefix.upper()} F1-score:  {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"], zero_division=0))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name_prefix.upper()}")

    cm_filename = f"confusion_matrix_{model_name_prefix}.png"
    plt.tight_layout()
    plt.savefig(cm_filename, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_filename}")

    # ---- ROC Curve (only if model supports decision_function or predict_proba) ----
    try:
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X)
        else:
            y_scores = None

        if y_scores is not None:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name_prefix.upper()}")
            plt.legend(loc="lower right")

            roc_filename = f"roc_curve_{model_name_prefix}.png"
            plt.tight_layout()
            plt.savefig(roc_filename, dpi=300)
            plt.close()
            print(f"Saved ROC curve to {roc_filename}")
        else:
            print(f"{model_name_prefix.upper()} does not support probability scores; ROC not generated.")
    except Exception as e:
        print(f"Error while plotting ROC for {model_name_prefix}: {e}")

    return acc, prec, rec, f1


# -------------------------
# 4. FEATURE IMPORTANCE PLOT (TF-IDF WEIGHTS) - LOGREG ONLY
# -------------------------

def plot_feature_importance(logreg_model, logreg_vectorizer, top_n=20):
    """
    Plots top positive and negative TF-IDF features learned by Logistic Regression.
    """
    print("\nGenerating feature importance plot for Logistic Regression...")

    feature_names = np.array(logreg_vectorizer.get_feature_names_out())
    # coef_ shape: (1, n_features) for binary classification
    coefs = logreg_model.coef_[0]

    # Top positive and negative coefficients
    top_pos_idx = np.argsort(coefs)[-top_n:]
    top_neg_idx = np.argsort(coefs)[:top_n]

    top_pos_words = feature_names[top_pos_idx]
    top_pos_weights = coefs[top_pos_idx]

    top_neg_words = feature_names[top_neg_idx]
    top_neg_weights = coefs[top_neg_idx]

    plt.figure(figsize=(10, 6))

    # Negative on left (red), positive on right (green)
    plt.barh(range(top_n), top_neg_weights, color="red", alpha=0.7, label="Negative")
    plt.yticks(range(top_n), top_neg_words)
    plt.xlabel("Coefficient Weight")
    plt.title("Top Negative Features (Logistic Regression)")
    plt.tight_layout()
    plt.savefig("feature_importance_negative_logreg.png", dpi=300)
    plt.close()
    print("Saved negative feature importance to feature_importance_negative_logreg.png")

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_pos_weights, color="green", alpha=0.7, label="Positive")
    plt.yticks(range(top_n), top_pos_words)
    plt.xlabel("Coefficient Weight")
    plt.title("Top Positive Features (Logistic Regression)")
    plt.tight_layout()
    plt.savefig("feature_importance_positive_logreg.png", dpi=300)
    plt.close()
    print("Saved positive feature importance to feature_importance_positive_logreg.png")


# -------------------------
# 5. MODEL COMPARISON BAR CHART
# -------------------------

def plot_model_comparison(metrics_dict):
    """
    metrics_dict = {
        "LogReg": {"acc": ..., "f1": ...},
        "SVM": {"acc": ..., "f1": ...}
    }
    """
    print("\nGenerating model comparison chart...")

    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]["acc"] for m in models]
    f1_scores = [metrics_dict[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, accuracies, width, label="Accuracy")
    plt.bar(x + width/2, f1_scores, width, label="F1-score")

    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)
    plt.close()
    print("Saved model comparison chart to model_comparison.png")


# -------------------------
# 6. MAIN EXECUTION
# -------------------------

if __name__ == "__main__":
    # 1. Load data
    df = load_sentiment140("training.1600000.processed.noemoticon.csv", sample_size=50000)

    # 2. Load models
    logreg_model, logreg_vectorizer, svm_model, svm_vectorizer = load_models()

    # 3. Evaluate Logistic Regression
    logreg_acc, logreg_prec, logreg_rec, logreg_f1 = evaluate_and_plot(
        df, logreg_model, logreg_vectorizer, model_name_prefix="logreg"
    )

    # 4. Evaluate SVM
    svm_acc, svm_prec, svm_rec, svm_f1 = evaluate_and_plot(
        df, svm_model, svm_vectorizer, model_name_prefix="svm"
    )

    # 5. Feature importance for Logistic Regression
    plot_feature_importance(logreg_model, logreg_vectorizer, top_n=20)

    # 6. Model comparison
    metrics = {
        "LogReg": {"acc": logreg_acc, "f1": logreg_f1},
        "SVM": {"acc": svm_acc, "f1": svm_f1}
    }
    plot_model_comparison(metrics)

    print("\nAll evaluation plots generated successfully.")
