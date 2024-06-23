def retrieve_logged_polls():
    # Retrieve logged polls from Firestore
    return [
        {"poll_text": "Who is the best actor?", "category": "movies"},
        {"poll_text": "What is the best programming language?", "category": "technology"},
    ]
