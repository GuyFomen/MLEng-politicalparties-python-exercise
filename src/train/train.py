import sys
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import numpy as np

sys.path.append('src')
from text_loader import loader

MLFLOW_TRACKING_URI = "data"
EXPERIMENT_NAME = "tweet-classifier"
MODEL_OUTPUT_PATH = "data/models"

def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and clean data
    data_loader = loader.DataLoader(filepath="data/Tweets.csv")
    data_loader.data.Tweet = data_loader.data.Tweet.apply(data_loader.clean_text)
    data_loader.data.Party = data_loader.data.Party.apply(data_loader.clean_text)

    # Drop classes with fewer than 2 samples (can't stratify-split them)
    counts = data_loader.data.Party.value_counts()
    data_loader.data = data_loader.data[data_loader.data.Party.isin(counts[counts >= 2].index)]

    X = data_loader.data.Tweet.values
    y_raw = data_loader.data.Party.values


    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Classes: {encoder.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline bundles vectorizer + model — fixes train/serve skew
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8)),
        ("clf", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=encoder.classes_))

        mlflow.log_param("tfidf_max_features", 2500)
        mlflow.log_param("xgb_n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        run_id = run.info.run_id
        print(f"\nMLflow run_id: {run_id}")
        print(f"Set MODEL_URI in main.py to:  runs:/{run_id}/model")

        # Save run_id to file for reference
        with open(f"{MODEL_OUTPUT_PATH}/run_id.txt", "w") as f:
            f.write(run_id)

    return run_id


if __name__ == "__main__":
    train()
