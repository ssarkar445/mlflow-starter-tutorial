import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn import datasets
from sklearn import model_selection
from sklearn import ensemble

seed = 2024
if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print(f"Name:{experiment.name}")

    with mlflow.start_run(run_name="logging_models",experiment_id=experiment.experiment_id) as myrun:

        X,y = datasets.make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=5,
            random_state=seed
            )
        X_train,X_test,y_train,y_test = model_selection.train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed
        )

        # Log Model using autologging specific flavour
        mlflow.sklearn.autolog()

        clf = ensemble.RandomForestClassifier(
            n_estimators=100,
            random_state=seed
        )
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)

        # Manual logging model depends on flavour

        mlflow.sklearn.log_model(sk_model=clf,artifact_path="random_forest_classifier")

        # Print Info about run
        print(f"Run ID:{myrun.info.run_id}")
        print(f"Experimrnt ID:{myrun.info.experiment_id}")
        print(f"Status:{myrun.info.status}")
        print(f"Start Time:{myrun.info.start_time}")
        print(f"End Time:{myrun.info.end_time}")
        print(f"Lifecycle Stage:{myrun.info.lifecycle_stage}")