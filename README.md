# Detecting Toxic Speech Patterns in Multiplayer Game Chats

## Honors Thesis - CS4490Z
**Author:** Dylan Sta Ana  
**Supervisor:** Dr. Umar Rehman  
**Course Instructor:** Prof. Nazim Madhavji  
**Department of Computer Science, Western University**

---

## Overview

This research investigates the effectiveness of Natural Language Processing (NLP) techniques for detecting toxic speech in multiplayer game chat environments. The study combines model-based toxicity classification with linguistic pattern analysis to understand how toxicity manifests in online game communication.

**Research Question:** *"How effectively can NLP-based classifiers detect toxic speech in multiplayer chat data, and what specific linguistic patterns characterize toxic behavior in online games?"*

---

## Thesis Objectives

| Objective | Description | Stage |
|-----------|-------------|-------|
| O1 | Identify and select appropriate public dataset | Stage 1 |
| O2 | Apply pre-trained NLP models and evaluate performance | Stage 2 |
| O3 | Conduct data analysis to identify linguistic and statistical patterns | Stage 3 |
| O4 | Visualize toxicity trends and classification outcomes | Stage 3 |
| O5 | Propose lightweight intervention strategies | Thesis Paper |


## Pipeline Overview

### Stage 1: Data Preprocessing & Failure Tag Detection
- Loads and cleans the Jigsaw Toxic Comment dataset
- Detects failure phenomena in original text:
  - All caps usage
  - Repeated letters
  - Excessive punctuation
  - Slang terms
- Saves processed data with tags for analysis

**Output:** `stage1_output.csv` with cleaned text and failure tags

### Stage 2: Model Evaluation
Compares two approaches to toxicity detection:
- **Traditional ML:** TF-IDF + Logistic Regression
- **Deep Learning:** BERT-based transformer model (`unitary/toxic-bert`)

Evaluates performance using:
- Accuracy, Precision, Recall, F1 Score
- Confusion matrices
- Per-tag performance metrics

**Output:** Model comparison metrics, predictions, confusion matrices

### Stage 3: Linguistic Pattern Analysis
Analyzes linguistic characteristics of toxic vs non-toxic comments:
- Message length patterns
- Vocabulary diversity (Type-Token Ratio)
- Profanity usage and types
- Pronoun patterns (I, you, we, they)
- Common phrases (bigrams and trigrams)
- Word cloud visualizations

**Output:** Comprehensive dashboard, word clouds, statistical measurements

---

## Key Findings

### Model Performance
| Metric | TF-IDF | BERT |
|--------|--------|------|
| Accuracy | 0.934 | 0.983 |
| Precision | 0.627 | 0.921 |
| Recall | 0.862 | 0.911 |
| F1 Score | 0.726 | 0.916 |

BERT significantly outperforms TF-IDF, reducing false positives from **1,661 to 252**.

### Linguistic Patterns
| Pattern | Toxic Comments | Non-Toxic Comments |
|---------|----------------|-------------------|
| Avg. Length | 52 words | 68 words |
| Profanity Presence | 44.0% | 5.9% |
| "You" usage | 68.9% | 49.6% |
| "We" usage | 6.6% | 14.3% |

### Common Toxic Phrases
- "nigger nigger", "fuck fuck", "faggot faggot"
- "super gay", "hate youi"
- Repetitive slur patterns

### Common Non-Toxic Phrases
- "talk page", "lol lol", "speedy deletion"
- "feel free", "welcome wikipedia thank"
- Collaborative and discussion-oriented language

---

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone the repository**
  git clone https://github.com/yourusername/Detecting-Toxic-Speech-Patterns-In-Game-Chats.git
  cd Detecting-Toxic-Speech-Patterns-In-Game-Chats

2. **Create VENV**
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activatE

3. Install Dependencies
  pip install -r requirements.txt

4. Download Dataset
  Place the Jigsaw Toxic Comment dataset (train.csv) in the data/ folder
  Dataset available from Kaggle

