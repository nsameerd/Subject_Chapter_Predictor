# Chapter Tagger for Educational Content

This project provides a system for training machine learning models to tag educational text content with relevant chapters or units based on subject matter. It includes scripts for both training the models and using them for prediction.




## Project Structure

- `train_models.py`: Script to train machine learning models for each subject.
- `predict.py`: Script to load trained models and predict chapter tags for new text.
- `data/`: Directory to store training data (e.g., `physics_chapter_tags.csv`).
- `models/`: Directory to store trained models (e.g., `physics_model.pkl`).




## Setup

To set up the project, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    pip install pandas scikit-learn joblib
    ```

3.  **Prepare data:**

    Place your training data CSV files (e.g., `physics_chapter_tags.csv`, `chemistry_chapter_tags.csv`) in the `data/` directory. Each CSV file should have at least two columns: `text` (containing the content to be classified) and `unit` (containing the corresponding chapter/unit tag).




## Usage

### Training Models

To train the models for all subjects, run the `train_models.py` script:

```bash
python train_models.py
```

This script will:
- Read the training data from the `data/` directory.
- Train a `CountVectorizer`, `TfidfTransformer`, and `SVC` (Support Vector Classifier) pipeline for each subject.
- Save the trained models as `.pkl` files in the `models/` directory.




### Making Predictions

To use the trained models for predicting chapter tags, you can use the `predict.py` script. The `ChapterTagger` class handles loading the models and making predictions.

```python
from predict import ChapterTagger

tagger = ChapterTagger()

test_subject = 'physics'
test_text = "Calculate the force required to accelerate a 2 kg object at 3 m/s²"

try:
    prediction = tagger.predict(test_subject, test_text)
    print(f"Predicted chapter for '{test_text}': {prediction}")
except ValueError as e:
    print(e)
```

**Note:** Ensure that the models for the desired subjects are present in the `models/` directory before attempting to make predictions.



