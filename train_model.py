### train_model.py
import pickle
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Enable MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris_Classification")


# Train model and log with MLflow
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model
    mlflow.sklearn.log_model(model, "iris_model")
    mlflow.log_params({"n_estimators": 100, "random_state": 42})
    print("Model logged successfully in MLflow.")