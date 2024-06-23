import functions_framework
import joblib
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv

from utils.firebase_utils import retrieve_logged_polls
from utils.preprocessing import preprocess_text
from utils.model_utils import load_or_create_model, upload_model_to_gcs

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

@functions_framework.http
def trainModel(request):
    try:
        data = retrieve_logged_polls()

        df = pd.DataFrame(data)

        df['poll_text_processed'] = df['poll_text'].apply(preprocess_text)

        X = df['poll_text_processed']
        y = df['category']

        # Load existing model or create a new one if it doesn't exist
        model_path = '/tmp/model_pipeline.pkl'
        pipeline = load_or_create_model(model_path, X, y)

        # Update the model with new data if the model was found
        if 'clf' in pipeline.named_steps:
            pipeline.named_steps['clf'].partial_fit(
                pipeline.named_steps['tfidf'].transform(X), y, classes=np.unique(y)
            )

        # Save the updated model
        joblib.dump(pipeline, model_path)

        # Upload the updated model to Google Cloud Storage
        upload_model_to_gcs(model_path)

        return 'Model retrained and uploaded successfully', 200
    except Exception as e:
        logging.error(f"Error retraining model: {e}")
        return str(e), 500
