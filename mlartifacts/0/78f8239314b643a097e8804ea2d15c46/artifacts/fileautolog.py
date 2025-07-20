from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load classification dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Try adjusting these to see accuracy variation
max_depth = 10
n_estimators = 50
mlflow.autolog()

with mlflow.start_run():
    # Train classifier
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log with MLflow
    # mlflow.log_param("Max_Depth", max_depth)
    # mlflow.log_param("N_Estimators", n_estimators)
    # mlflow.log_param("Random_State", random_state)
    # mlflow.log_metric("Accuracy", accuracy)
    
    cm= confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.title("Confusion Matrix")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("Confusion_Matrix.png")
     
    mlflow.log_artifact(__file__)
    mlflow.set_tags({"Author": "Roshan Singh", "Experiment": "Breast Cancer Classification"})
     

    print(f"Accuracy: {accuracy:.4f}")
