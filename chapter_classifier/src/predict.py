import os
import joblib
from sklearn.pipeline import Pipeline

# Configuration
SUBJECTS = ['physics', 'chemistry', 'maths', 'biology']
MODELS_DIR = '../models'

class ChapterTagger:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models from disk"""
        for subject in SUBJECTS:
            model_path = f'{MODELS_DIR}/{subject}_model.pkl'
            if os.path.exists(model_path):
                self.models[subject] = joblib.load(model_path)
            else:
                print(f"Warning: Model not found for {subject}")
    
    def predict(self, subject: str, text: str) -> str:
        """
        Predict the chapter/unit for a given subject and text
        
        Args:
            subject: One of 'physics', 'chemistry', 'maths', or 'biology'
            text: The input text to classify
            
        Returns:
            Predicted chapter/unit label
        """
        if subject not in self.models:
            raise ValueError(f"No model available for subject: {subject}")
        
        return self.models[subject].predict([text])[0]

if __name__ == '__main__':
    # Example usage
    tagger = ChapterTagger()
    
    # Test prediction
    test_subject = 'physics'
    test_text = "Calculate the force required to accelerate a 2 kg object at 3 m/s²"
    
    try:
        prediction = tagger.predict(test_subject, test_text)
        print(f"Predicted chapter for '{test_text}': {prediction}")
    except ValueError as e:
        print(e)