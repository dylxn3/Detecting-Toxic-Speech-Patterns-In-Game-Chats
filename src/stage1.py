# stage1.py

import pandas as pd
import re
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
    Detects common failure phenomena in ORIGINAL text (before cleaning)
    Returns a list of tags.
    """
    tags = []

    # Expanded slang list for game chat
    slang_terms = ["lol", "lmao", "rofl", "wtf", "omg", 
                   "haha", "xd", "brb", "gg", "noob", "ez", "rekt"]

    # Regex for repeated letters (3 or more consecutive same letters)
    repeated_letters = re.compile(r'(.)\1{3,}')

    # Detect slang terms (case-insensitive)
    text_lower = text.lower()
    if any(re.search(rf'\b{slang}\b', text_lower) for slang in slang_terms):
        tags.append("slang")

    # Detect repeated letters (works on original text)
    if repeated_letters.search(text):
        tags.append("repeated_letters")

    # All caps (check ORIGINAL text BEFORE lowercasing)
    if text.isupper() and len(text) > 3:
        tags.append("all_caps")

    # Excessive punctuation (check ORIGINAL text BEFORE removal)
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

    print(f"Loaded {len(df)} comments from {input_file}")
    
    # ============================================================
    # FIXED: Proper toxicity column detection
    # ============================================================
    print("\n" + "="*60)
    print("TOXICITY COLUMN DETECTION")
    print("="*60)
    
    print(f"\nAvailable columns in dataset: {list(df.columns)}")

    # Define expected toxicity columns
    expected_toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # Find which of these columns actually exist in the dataframe
    toxic_columns = []
    for col in expected_toxic_cols:
        if col in df.columns:
            toxic_columns.append(col)
            print(f"✓ Found toxicity column: '{col}'")
        else:
            # Try case-insensitive match
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                toxic_columns.append(matches[0])
                print(f"✓ Found toxicity column (case-insensitive): '{matches[0]}'")
            else:
                print(f"✗ Expected column '{col}' not found - creating with 0s")
                df[col] = 0
                toxic_columns.append(col)

    # If no toxicity columns found, show error
    if not toxic_columns or all(df[col].sum() == 0 for col in toxic_columns):
        print("\n⚠️  CRITICAL WARNING: No toxicity columns found or all are zero!")
        print("Check your dataset columns. This will result in all comments being labeled as non-toxic.")
        print("Available columns:", list(df.columns))
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)

    # Aggregate toxicity into one column (1 if any toxicity type is present)
    df["toxic_label"] = df[toxic_columns].max(axis=1)
    
    # Print toxicity distribution
    toxic_count = df["toxic_label"].sum()
    print(f"\n✅ Toxicity distribution:")
    print(f"  Toxic (1): {toxic_count:,} ({toxic_count/len(df)*100:.2f}%)")
    print(f"  Non-toxic (0): {len(df)-toxic_count:,} ({(len(df)-toxic_count)/len(df)*100:.2f}%)")

    # Show breakdown by toxicity type
    print(f"\n📊 Breakdown by toxicity type:")
    for col in toxic_columns:
        if col in df.columns:
            count = df[col].sum()
            print(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")

    # CRITICAL FIX: Detect tags on ORIGINAL text BEFORE cleaning
    text_column = "comment_text"
    if text_column not in df.columns:
        print(f"Error: No '{text_column}' column found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print("\nDetecting failure phenomena on ORIGINAL text...")
    df['failure_tags'] = df[text_column].apply(tag_failure_phenomena)

    # THEN clean text for model input
    print("Cleaning text for model input...")
    df['cleaned_text'] = df[text_column].apply(clean_text)

    # Show detected tags
    all_tags = []
    for tags in df['failure_tags']:
        all_tags.extend(tags)
    
    tag_counts = Counter(all_tags)
    print("\nFailure phenomena detected:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count} ({count/len(df)*100:.2f}%)")

    # Create output directory
    output_folder = "../output/stage1_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Save main data file
    output_file = os.path.join(output_folder, "stage1_output.csv")
    df.to_csv(output_file, index=False)
    print(f"\nStage 1 data saved to: {output_file}")

    # Save tag statistics
    tag_stats = pd.DataFrame({
        'tag': list(tag_counts.keys()),
        'count': list(tag_counts.values()),
        'percentage': [count/len(df)*100 for count in tag_counts.values()]
    }).sort_values('count', ascending=False)
    
    tag_stats_file = os.path.join(output_folder, "tag_statistics.csv")
    tag_stats.to_csv(tag_stats_file, index=False)
    print(f"Tag statistics saved to: {tag_stats_file}")

    # Generate graphs
    print("\nGenerating visualizations...")
    sns.set_style("whitegrid")

    # 1. Toxic label distribution
    plt.figure(figsize=(8,5))
    ax = sns.countplot(x='toxic_label', data=df)
    plt.title("Distribution of Toxic Comments", fontsize=14, fontweight='bold')
    plt.xlabel("Toxic Label (0 = Non-toxic, 1 = Toxic)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Add count labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()), 
                   ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "toxic_label_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Failure tags distribution
    if all_tags:
        plt.figure(figsize=(10,6))
        tag_counts_series = pd.Series(all_tags).value_counts()
        
        ax = sns.barplot(x=tag_counts_series.index, y=tag_counts_series.values)
        plt.xticks(rotation=45, ha='right')
        plt.title("Failure Phenomena Distribution", fontsize=14, fontweight='bold')
        plt.ylabel("Count", fontsize=12)
        plt.xlabel("Phenomenon Type", fontsize=12)
        
        # Add count labels on bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "failure_tags_distribution.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Phenomena by toxicity (stacked bar chart)
    if all_tags:
        # Create a dataframe of tags by toxicity
        tag_by_toxicity = []
        for _, row in df.iterrows():
            for tag in row['failure_tags']:
                tag_by_toxicity.append({
                    'tag': tag,
                    'is_toxic': row['toxic_label']
                })
        
        tag_df = pd.DataFrame(tag_by_toxicity)
        tag_crosstab = pd.crosstab(tag_df['tag'], tag_df['is_toxic'])
        tag_crosstab.columns = ['Non-toxic', 'Toxic']
        
        # Plot
        plt.figure(figsize=(10,6))
        tag_crosstab.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title("Failure Phenomena by Toxicity", fontsize=14, fontweight='bold')
        plt.xlabel("Phenomenon Type", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Message Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "phenomena_by_toxicity.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Save a summary report
    with open(os.path.join(output_folder, "summary_report.txt"), 'w') as f:
        f.write("STAGE 1 PROCESSING SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total comments processed: {len(df):,}\n")
        f.write(f"Toxic comments: {toxic_count:,} ({toxic_count/len(df)*100:.2f}%)\n")
        f.write(f"Non-toxic comments: {len(df)-toxic_count:,} ({(len(df)-toxic_count)/len(df)*100:.2f}%)\n\n")
        
        f.write("BREAKDOWN BY TOXICITY TYPE:\n")
        f.write("-"*30 + "\n")
        for col in toxic_columns:
            if col in df.columns:
                count = df[col].sum()
                f.write(f"{col}: {count:,} ({count/len(df)*100:.2f}%)\n")
        
        f.write("\nFAILURE PHENOMENA DETECTED:\n")
        f.write("-"*30 + "\n")
        for tag, count in tag_counts.most_common():
            f.write(f"{tag}: {count:,} ({count/len(df)*100:.2f}%)\n")
        
        f.write("\nFILES GENERATED:\n")
        f.write("-"*30 + "\n")
        f.write("stage1_output.csv - Main data file with cleaned text and tags\n")
        f.write("tag_statistics.csv - Statistics for each failure phenomenon\n")
        f.write("toxic_label_distribution.png - Bar chart of toxic vs non-toxic\n")
        f.write("failure_tags_distribution.png - Distribution of all failure tags\n")
        f.write("phenomena_by_toxicity.png - Stacked bar chart of tags by toxicity\n")
        f.write("summary_report.txt - This summary file\n")

    print(f"\nAll outputs saved to: {output_folder}")
    print("\nFiles generated:")
    print(f"  • stage1_output.csv - Main data file")
    print(f"  • tag_statistics.csv - Tag statistics")
    print(f"  • toxic_label_distribution.png - Toxicity distribution chart")
    print(f"  • failure_tags_distribution.png - Tags distribution chart")
    print(f"  • phenomena_by_toxicity.png - Tags by toxicity chart")
    print(f"  • summary_report.txt - Text summary")
    
    print("\n" + "="*60)
    print("STAGE 1 EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)

# Run the pipeline
if __name__ == "__main__":
    main()