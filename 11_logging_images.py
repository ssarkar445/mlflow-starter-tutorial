import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

seed=2024

if __name__=="__main__":
    
    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print(f"Name:{experiment.name}")

    with mlflow.start_run(run_name="logging_images",experiment_id=experiment.experiment_id) as myrun:

        X,y = make_classification(n_samples=1000,n_features=10,n_informative=5,n_redundant=5,random_state=seed)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed)

        rfc = RandomForestClassifier(n_estimators=100,random_state=seed)
        rfc.fit(X_train,y_train)
        y_pred = rfc.predict(X_test)

        # Logging precision recall curve
        fig_pr = plt.figure()
        pr_display = PrecisionRecallDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        plt.title("Precision-Recall Curve")
        plt.legend()

        mlflow.log_figure(fig_pr,"metrics/precision-recall-curve.png")


        # Logging the roc
        fig_roc = plt.figure()
        pr_display = RocCurveDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        plt.title("ROC Curve")
        plt.legend()

        mlflow.log_figure(fig_roc,"metrics/roc-curve.png")


        # Logging confusion matrix
        fig_cm = plt.figure()
        pr_display = ConfusionMatrixDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        plt.title("Confusion Matrix")
        plt.legend()

        mlflow.log_figure(fig_cm,"metrics/confusion-matrix.png")

        # Print Info about run
        print(f"Run ID:{myrun.info.run_id}")
        print(f"Experimrnt ID:{myrun.info.experiment_id}")
        print(f"Status:{myrun.info.status}")
        print(f"Start Time:{myrun.info.start_time}")
        print(f"End Time:{myrun.info.end_time}")
        print(f"Lifecycle Stage:{myrun.info.lifecycle_stage}")