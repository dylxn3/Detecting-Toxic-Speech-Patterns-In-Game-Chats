# preprocessing.py
import re

# Text cleaning function
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Failure-tagging function
def tag_failure_phenomena(text):
    slang_terms = ["lol", "lmao", "rofl", "wtf", "omg", "idk", "smh", "brb", "btw", "stfu"]
    reclaimed_terms = ["bitch", "queer", "nigga", "gay", "slut"]
    coded_insults = ["go back", "trash", "loser", "weak", "pathetic", "idiot"]
    sarcasm_indicators = ["yeah right", "as if", "sure", "totally", "of course", "great job", "nice one"]

    text_lower = text.lower()
    tags = []
    if any(word in text_lower.split() for word in slang_terms):
        tags.append("slang")
    if any(word in text_lower.split() for word in reclaimed_terms):
        tags.append("reclaimed")
    if any(word in text_lower for word in coded_insults):
        tags.append("coded_insult")
    if any(phrase in text_lower for phrase in sarcasm_indicators):
        tags.append("sarcasm")
    if len(tags) == 0:
        tags.append("none")
    return ",".join(tags)
