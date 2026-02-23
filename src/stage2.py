# Stage 2 trains models to detect toxicity and measures how well they perform 
# both overall and under linguistic distortions.

# note: SARCASM failure tag is added in stage 2 as 
# adding in preprocessing (stage 1) will cause skewed info 
# needs HUGGING FACE to work efficently based on context 

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import pipeline
import torch

# Extra failure tag - Sarcasm 
def detect_sarcasm(text, sarcasm_classifier=None):
    """Return True if sarcasm detected"""
    if sarcasm_classifier is None:
        return False
    result = sarcasm_classifier(text)[0]
    return result["label"].lower() == "sarcasm"


# Main function for stage 2 
def main():
    # Load Stage 1 output
    input_file = "../output/stage1_output.csv"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Make sure Stage 1 is complete.")
        sys.exit(1)

    # Adding sarcasm to failure tags
    use_sarcasm = True
    sarcasm_classifier = None
    if use_sarcasm:
        try:
            sarcasm_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sarcasm",
                device=0 if torch.cuda.is_available() else -1
            )
            df['failure_tags'] = df.apply(
                lambda row: row['failure_tags'] + ["sarcasm"] if detect_sarcasm(row['cleaned_text'], sarcasm_classifier) else row['failure_tags'],
                axis=1
            )
        except Exception as e:
            print("Sarcasm classifier failed:", e)
            print("Continuing without sarcasm detection.")
            use_sarcasm = False

    X = df['cleaned_text']
    y = df['toxic_label']

    X_train, X_test, y_train, y_test, train_tags, test_tags = train_test_split(
        X, y, df['failure_tags'], test_size=0.2, random_state=42
    )

    # Model 1 - TF - IDF (Baseline)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)

    # Model 2 - BERT via Hugging Face
    bert_model = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=0 if torch.cuda.is_available() else -1
    )
    # Hugging Face pipeline maps to 0/1
    def predict_bert(texts):
        preds = []
        batch_size = 16  # reduce memory footprint
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = bert_model(batch)
            for r in results:
                label = r['label']
                if label in ["LABEL_0", "NOT_TOXIC"]:
                    preds.append(0)
                else:
                    preds.append(1)
        return np.array(preds)

    y_pred_bert = predict_bert(list(X_test))

    
    # Compute metrics
    
    def compute_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    overall_metrics = {
        "TF-IDF": compute_metrics(y_test, y_pred_lr),
        "BERT": compute_metrics(y_test, y_pred_bert)
    }

    print("Overall Metrics:")
    print(pd.DataFrame(overall_metrics))

    
    # Metrics per failure phenomenon
    phenomena = ["slang", "repeated_letters", "all_caps", "excessive_punctuation"]
    if use_sarcasm:
        phenomena.append("sarcasm")

    per_tag_metrics = {}
    for tag in phenomena:
        indices = test_tags.apply(lambda tags: tag in tags)
        if indices.sum() == 0:
            continue
        per_tag_metrics[tag] = {
            "TF-IDF": compute_metrics(y_test[indices], y_pred_lr[indices]),
            "BERT": compute_metrics(y_test[indices], y_pred_bert[indices])
        }

    print("\nMetrics Per Failure Phenomenon:")
    for tag, metrics in per_tag_metrics.items():
        print(f"\nTag: {tag}")
        print(pd.DataFrame(metrics))

    # Save misclassified examples for Stage 3
    output_folder = "../output"
    os.makedirs(output_folder, exist_ok=True)

    df_test = pd.DataFrame({
        "text": X_test,
        "true_label": y_test,
        "pred_lr": y_pred_lr,
        "pred_bert": y_pred_bert,
        "failure_tags": test_tags
    })
    # Add misclassification flags
    df_test['misclassified_lr'] = df_test['true_label'] != df_test['pred_lr']
    df_test['misclassified_bert'] = df_test['true_label'] != df_test['pred_bert']

    df_test.to_csv(os.path.join(output_folder, "stage2_predictions.csv"), index=False)
    print(f"\nStage 2 complete. Predictions and misclassified examples saved to {output_folder}")

    # Generate graphs
    sns.set(style="whitegrid")

    # Overall model comparison
    metrics_df = pd.DataFrame({
        "TF-IDF": [overall_metrics["TF-IDF"]["f1"], overall_metrics["TF-IDF"]["precision"], overall_metrics["TF-IDF"]["recall"]],
        "BERT": [overall_metrics["BERT"]["f1"], overall_metrics["BERT"]["precision"], overall_metrics["BERT"]["recall"]]
    }, index=["F1", "Precision", "Recall"])

    metrics_df.plot(kind="bar", figsize=(8,5))
    plt.title("Overall Model Performance")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "model_overall_metrics.png"))
    plt.close()

    # Per-failure phenomenon F1 comparison
    per_tag_f1 = pd.DataFrame({
        tag: {"TF-IDF": per_tag_metrics[tag]["TF-IDF"]["f1"], "BERT": per_tag_metrics[tag]["BERT"]["f1"]}
        for tag in per_tag_metrics
    }).T

    per_tag_f1.plot(kind="bar", figsize=(8,5))
    plt.title("F1 Score per Failure Phenomenon")
    plt.ylabel("F1 Score")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "f1_per_failure_tag.png"))
    plt.close()

    print("Graphs saved in the output folder.")

# Run pipeline
if __name__ == "__main__":
    main()