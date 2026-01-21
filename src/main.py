# main.py
import pandas as pd
from src.preprocessing import clean_text, tag_failure_phenomena

# Load dataset
df = pd.read_csv(r"data\train.csv")

# Clean text
df['clean_text'] = df['comment_text'].apply(clean_text)

# Tag failure-prone linguistic phenomena
df['failure_tags'] = df['clean_text'].apply(tag_failure_phenomena)

# Save the processed dataset
df.to_csv(r"outputs\tagged_toxic_comments.csv", index=False)
print("Dataset cleaned and tagged!")
