import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://localhost:5000')
# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
random_state=60

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=random_state)

# Define the params for RF model
max_depth = 10
n_estimators = 20

 

with mlflow.start_run():
 
  rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=random_state)
  rf.fit(X_train,y_train)
  y_pred=rf.predict(X_test)
  accuracy=accuracy_score(y_test,y_pred)

  mlflow.log_metric('Accuracy', accuracy)
  mlflow.log_param('Max_Depth',max_depth)
  mlflow.log_param('N_Estimator',n_estimators)
  mlflow.log_param('Random_State',random_state)
  mlflow.log_artifact(__file__)
  print("Accuracy=", accuracy)  

