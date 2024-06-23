import re

def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        raise ValueError(f"Error preprocessing text: {e}")
