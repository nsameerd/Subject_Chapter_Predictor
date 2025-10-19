import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib

# Configuration
SUBJECTS = ['physics', 'chemistry', 'maths', 'biology']
DATA_DIR = '../data'
MODELS_DIR = '../models'

def train_model_for_subject(subject):
    # Load the training data
    df = pd.read_csv(f'{DATA_DIR}/{subject}_chapter_tags.csv')
    
    # Separate features and labels
    X = df['text'].values
    y = df['unit'].values
    
    # Define pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC(kernel='poly'))
    ])
    
    # Train the pipeline
    pipeline.fit(X, y)
    
    # Save the trained model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, f'{MODELS_DIR}/{subject}_model.pkl')
    print(f"Model trained and saved for {subject}")

if __name__ == '__main__':
    for subject in SUBJECTS:
        train_model_for_subject(subject)