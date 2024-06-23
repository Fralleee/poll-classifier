import functions_framework
import joblib
import logging
import os
from dotenv import load_dotenv
from flask import jsonify

from utils.preprocessing import preprocess_text
from utils.keyword_utils import load_keywords, classify_by_keywords
from utils.model_utils import download_model

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

keywords_csv_path = 'data/keywords.csv'
keywords = load_keywords(keywords_csv_path)

project_id = os.getenv("GCP_PROJECT_ID")
bucket_name = os.getenv("GCP_BUCKET_NAME")
source_blob_name = os.getenv("MODEL_FILE_NAME")
destination_file_name = "/tmp/model_pipeline.pkl"

logging.debug(f"Project ID: {project_id}")
logging.debug(f"Bucket Name: {bucket_name}")
logging.debug(f"Model File Name: {source_blob_name}")

try:
    download_model(bucket_name, source_blob_name, destination_file_name)
    pipeline = joblib.load(destination_file_name)
    logging.debug("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    pipeline = None

@functions_framework.http
def classifyPoll(request):
    if pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        request_json = request.get_json(silent=True)
        poll_text = request_json.get('poll_text')
        poll_text_processed = preprocess_text(poll_text)

        # First attempt to classify by keywords
        category = classify_by_keywords(poll_text, keywords)

        # If no category is found by keywords, use the ML model
        if category is None:
            X_new = pipeline.named_steps['tfidf'].transform([poll_text_processed])
            category = pipeline.named_steps['clf'].predict(X_new)[0]

        logging.debug(f"Classification result: {category}")
        return jsonify({'category': category})
    except Exception as e:
        logging.error(f"Error in classify_poll: {e}")
        return jsonify({'error': str(e)}), 500
