import pandas as pd
import logging

def load_keywords(csv_path):
    try:
        df_keywords = pd.read_csv(csv_path)
        keywords = {}
        for index, row in df_keywords.iterrows():
            category = row['category']
            keyword = row['keyword']
            if category not in keywords:
                keywords[category] = []
            keywords[category].append(keyword)
        logging.debug("Keywords loaded successfully.")
        return keywords
    except Exception as e:
        logging.error(f"Error loading keywords: {e}")
        raise

def classify_by_keywords(question, keywords):
    try:
        question = question.lower()
        for category, keyword_list in keywords.items():
            for keyword in keyword_list:
                if keyword in question:
                    logging.debug(f"Classified by keyword: {category}")
                    return category
        logging.debug("No classification by keyword found.")
        return None
    except Exception as e:
        logging.error(f"Error in classify_by_keywords: {e}")
        raise
