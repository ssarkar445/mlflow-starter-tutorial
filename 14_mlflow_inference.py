import mlflow
from mlflow.models import infer_signature
from mlflow_utils import get_mlflow_experiment

from sklearn import datasets
from sklearn import model_selection
from sklearn import ensemble

from pathlib import Path

import pandas as pd

seed=2024

if __name__=="__main__":

    run_id = "ffde8d8dafd447d0b6b136e48c132083"

    X,y = datasets.make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=5,
            random_state=seed
            )
    
    X = pd.DataFrame(X,columns=[f"feature_{i}" for i in range(10)])
    y = pd.DataFrame(y,columns=['target'])

    _,X_test,_,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=seed)

    # load model
    model_uri = f'runs:/{run_id}/random_forest_classifier'
    
    local_model_uri = f'file:///C:/Drive_1/mlflow/testing_mlflow1_artifacts/{run_id}/artifacts/random_forest_classifier'

    rfc = mlflow.sklearn.load_model(model_uri=local_model_uri)

    y_pred = rfc.predict(X_test)
    y_pred = pd.DataFrame(y_pred,columns=['prediction'])

    print(y_pred.head())