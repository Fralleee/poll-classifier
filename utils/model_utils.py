import joblib
import pandas as pd
from google.cloud import storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import os
import logging
from utils.preprocessing import preprocess_text

def download_model(bucket_name, source_blob_name, destination_file_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.debug("Model downloaded successfully.")
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        raise

def load_or_create_model(model_path, X=None, y=None):
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        logging.debug("Existing model loaded.")
    else:
        # We need to retrain the model from scratch
        # Load sample data
        sample_data_path = 'data/sample_data.csv'
        sample_df = pd.read_csv(sample_data_path)
        sample_df['poll'] = sample_df['poll'].apply(preprocess_text)

        # Combine sample data with new input data
        if X is not None and y is not None:
            sample_X = sample_df['poll']
            sample_y = sample_df['category']
            X = pd.concat([X, sample_X])
            y = pd.concat([y, sample_y])
        else:
            X = sample_df['poll']
            y = sample_df['category']

        # Train a new model with combined data
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1))
        ])
        pipeline.fit(X, y)
        logging.debug("New model trained with combined data.")
    return pipeline

def upload_model_to_gcs(model_path):
    client = storage.Client()
    bucket = client.bucket(os.getenv('GCP_BUCKET_NAME'))
    blob = bucket.blob(os.getenv('MODEL_FILE_NAME'))
    blob.upload_from_filename(model_path)