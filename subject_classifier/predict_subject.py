import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Add your text preprocessing steps here
        return X

# Load the test data
test_df = pd.read_csv('subjects-questions.csv')
X_test = test_df['eng'].values[0:1000]
y_test = test_df['Subject'].values[0:1000]

# Define classifiers to compare (same as in training)
classifier_names = [
    'SVM (Poly)',
    'SVM (RBF)',
    'Logistic Regression',
    'Multinomial NB',
    'Random Forest',
    'k-NN'
]

results = []

# Evaluate each classifier on test data
for name in classifier_names:
    model_filename = f'model_{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.pkl'
    
    try:
        pipeline = joblib.load(model_filename)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((name, accuracy))
        print(f"{name} accuracy: {accuracy:.4f}")
    except FileNotFoundError:
        print(f"Model {name} not found. Skipping...")

# Display results
print("\nFinal Classifier Comparison on Test Data:")
for name, accuracy in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{name:20} {accuracy:.4f}")