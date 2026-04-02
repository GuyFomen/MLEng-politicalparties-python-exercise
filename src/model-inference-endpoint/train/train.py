import sys
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

sys.path.append('src')
from text_loader import loader

MLFLOWW_TRACKING_URI = "data"
EXPERIMENT_NAME = "tweet-classifer"
MODEL_OUTPUT_PATH = "data/models"

def train():
    mlflow.set_tracking_uri(MLFLOWW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_loader = loader.DataLoader()
    data_loader.data.Tweets = data_loader.data.Tweet.apply(data_loader.clean_text)
    data_loader.data.Party = data_loader.data.Party.apply(data_loader.clean_text)


    # drop classe with fewer that 2 samples. can't split them
    counts = data_loader.data.Party.value_counts()
    data_loader.data = data_loader.data[data_loader.data.Party.isin(counts[counts >=2].index)]

    X = data_loader.data.Tweet.values
    y_raw = data_loader.data.Party.values

    y = data_loader.label_encoder(y_raw)
    print(f"Classes: {data_loader.encoder.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8)), 
    ("clf", XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42))])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=data_loader.encoder.classes_))

        mlflow.log_param("xgb n_estimators", 100)
        mlflow.log_param("tfidf_max_features", 2500)
        mlflow.log_metric("accuracy", acc)

        #save the model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        model_path = f"data/{experiment_id}/{run_id}/artifacts/model"
        print(f"\nMLFLOW run_id:{run_id}")
        print(f"Model path: {model_path}")

        # Save run_id and portable model path for reference
        with open(f"{MODEL_OUTPUT_PATH}/run_id.txt", "w") as f:
            f.write(run_id)
        with open(f"{MODEL_OUTPUT_PATH}/model_path.txt", "w") as f:
            f.write(model_path)
    return run_id
if __name__ == "__main__":
    train()