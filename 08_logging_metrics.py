import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")

    print(f"Name:{experiment.experiment_id}")

    with mlflow.start_run(run_name="logging_metrics",experiment_id=experiment.experiment_id) as myrun:

        # your machine learning code goes here

        mlflow.log_metric("mse",0.01)

        metrics = {
            "mse":0.01,
            "mae":0.02,
            "rmse":0.03,
            "r2":0.04,
            "log_loss":0.05
        }

        mlflow.log_metrics(metrics)

        print(f"Run ID:{myrun.info.run_id}")
        print(f"Experimrnt ID:{myrun.info.experiment_id}")
        print(f"Status:{myrun.info.status}")
        print(f"Start Time:{myrun.info.start_time}")
        print(f"End Time:{myrun.info.end_time}")
        print(f"Lifecycle Stage:{myrun.info.lifecycle_stage}")

