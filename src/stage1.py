# stage1.py

import pandas as pd
import re
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing functions
def clean_text(text):
    """Lowercase, strip whitespace, remove newlines and special characters"""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def tag_failure_phenomena(text):
    """
    Detects common failure phenomena in text:
    - Slang words
    - Repeated letters
    - All caps messages
    - Excessive punctuation
    Returns a list of tags.
    """
    tags = []

    # Expanded slang list for game chat
    slang_terms = ["lol", "lmao", "rofl", "wtf", "omg", 
                   "haha", "xd", "brb", "gg", "noob", "ez", "rekt"]

    # Regex for repeated letters (4 or more consecutive letters)
    repeated_letters = re.compile(r'(.)\1{3,}')  # e.g., "heyyyy"

    # Detect slang terms (whole word match)
    for slang in slang_terms:
        if re.search(rf'\b{slang}\b', text):
            tags.append(f"slang:{slang}")

    # Detect repeated letters
    if repeated_letters.search(text):
        tags.append("repeated_letters")

    # All caps (longer than 3 chars)
    if text.isupper() and len(text) > 3:
        tags.append("all_caps")

    # Excessive punctuation
    if re.search(r'[!?.]{3,}', text):
        tags.append("excessive_punctuation")

    return tags

# Main Stage 1 pipeline
def main():
    # Load dataset
    input_file = "../data/train.csv"  # default path
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Make sure the file exists.")
        sys.exit(1)

    # Detect toxic columns (handle typo 'servere_toxic')
    expected_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    actual_cols = df.columns.str.lower().str.replace(' ', '_')

    toxic_columns = []
    for col in expected_cols:
        matches = [c for c in actual_cols if c.startswith(col[:4])]
        if matches:
            toxic_columns.append(matches[0])
        else:
            print(f"Warning: column '{col}' not found in CSV, adding it as 0s")
            df[col] = 0
            toxic_columns.append(col)

    # Aggregate toxicity into one column
    df["toxic_label"] = df[toxic_columns].max(axis=1)

    # Clean text and tag failures
    text_column = "comment_text"  # your CSV text column
    if text_column not in df.columns:
        print(f"Error: No '{text_column}' column found in CSV.")
        sys.exit(1)

    df['cleaned_text'] = df[text_column].apply(clean_text)
    df['failure_tags'] = df['cleaned_text'].apply(tag_failure_phenomena)

    # Save output to output/ folder
    output_folder = "../output"
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "stage1_output.csv")
    df.to_csv(output_file, index=False)
    print(f"Stage 1 complete. Output saved to {output_file}")

    # -----------------------------
    # Generate graphs
    # -----------------------------
    sns.set(style="whitegrid")

    # 1. Toxic label distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='toxic_label', data=df)
    plt.title("Distribution of Toxic Comments")
    plt.xlabel("Toxic Label (0 = Non-toxic, 1 = Toxic)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "toxic_label_distribution.png"))
    plt.close()

    # 2. Top failure tags
    all_tags = [tag for tags_list in df['failure_tags'] for tag in tags_list]
    tag_counts = pd.Series(all_tags).value_counts().head(10)  # top 10 tags

    plt.figure(figsize=(8,5))
    sns.barplot(x=tag_counts.index, y=tag_counts.values)
    plt.xticks(rotation=45)
    plt.title("Top Failure Tags in Game Chat")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "failure_tags_distribution.png"))
    plt.close()

    print("Graphs saved in the output folder.")

# Run the pipeline
if __name__ == "__main__":
    main()