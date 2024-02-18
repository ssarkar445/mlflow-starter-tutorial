import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")

    print(f"Name:{experiment.experiment_id}")

    with mlflow.start_run(run_name="logging_parameters",experiment_id=experiment.experiment_id) as myrun:

        # your machine learning code goes here

        mlflow.log_param("learning_rate",0.01)

        parameters = {
            "learning_rate":0.01,
            "epoches":10,
            "batch_size":100,
            "loss_function":"mse",
            "optimizer":"adam"
        }

        mlflow.log_params(parameters)

        print(f"Run ID:{myrun.info.run_id}")
        print(f"Experimrnt ID:{myrun.info.experiment_id}")
        print(f"Status:{myrun.info.status}")
        print(f"Start Time:{myrun.info.start_time}")
        print(f"End Time:{myrun.info.end_time}")
        print(f"Lifecycle Stage:{myrun.info.lifecycle_stage}")

