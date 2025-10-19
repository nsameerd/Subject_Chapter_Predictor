import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score

# Load the training data
df = pd.read_csv('subjects-questions.csv')

# Separate the features (questions) and the labels (subjects)
X = df['eng'].values
y = df['Subject'].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Transformer to preprocess text data
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Add your text preprocessing steps here
        return X

# Define classifiers to compare
classifiers = [
    ('SVM (Poly)', SVC(kernel='poly', C=1.0, degree=3, gamma='auto', cache_size=25000)),
    ('SVM (RBF)', SVC(kernel='rbf', C=1.0, gamma='auto', cache_size=25000)),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Multinomial NB', MultinomialNB()),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('k-NN', KNeighborsClassifier(n_neighbors=5))
]

results = []

# Train and evaluate each classifier
for name, classifier in classifiers:
    print(f"\nTraining {name}...")
    
    pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', classifier)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Save the trained model
    model_filename = f'model_{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.pkl'
    joblib.dump(pipeline, model_filename)
    
    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    results.append((name, accuracy))
    
    print(f"{name} accuracy: {accuracy:.4f}")

# Display results
print("\nClassifier Comparison:")
for name, accuracy in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{name:20} {accuracy:.4f}")