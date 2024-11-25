import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

exp_name = 'BreastCancer_Exp'
mlflow.set_experiment(exp_name)

n_estimators_opt = [50,100,150]
max_depth_opt = [3,5,7]
random_state=42

for n_estimators in n_estimators_opt:
    for max_depth in max_depth_opt:
        with mlflow.start_run(run_name=f"RF_Model_n{n_estimators}_d{max_depth}") as run:
            mlflow.log_param("n_estimators",n_estimators)
            mlflow.log_param("max_depth",max_depth)
            mlflow.log_param("random_state",random_state)

            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=random_state)
            model.fit(X_train,y_train)
            
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_pred=predictions,y_true=y_test)
            precision = precision_score(y_pred=predictions,y_true=y_test)
            recall = recall_score(y_pred=predictions,y_true=y_test)

            mlflow.log_metric('accuracy',accuracy)
            mlflow.log_metric('precision',precision)
            mlflow.log_metric('recall',recall)
            
            np.savetxt('feature_importances.csv',model.feature_importances_,delimiter=',')
            mlflow.log_artifact('feature_importances.csv')

            mlflow.sklearn.log_model(model,artifact_path='model')
            
            model_name = "BreastCancer_Exp"
            model_version = mlflow.register_model(f'runs:/{run.info.run_id}/model',model_name)

            print(model_name,model_version.version)
            print(f'Run ID: {run.info.run_id} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}')
