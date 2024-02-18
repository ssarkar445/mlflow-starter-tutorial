import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__=="__main__":

    experiment_id = create_mlflow_experiment(
        experiment_name="testing_mlflow1",
        artifact_location="testing_mlflow1_artifacts",
        tags={"env":"dev","version":"1.0.0"}
    )

    # mlflow.set_experiment(experiment_name="testing_mlflow1")

    with mlflow.start_run(run_name="testing",experiment_id=experiment_id) as myrun:

        # Your machine learning code goes here
        mlflow.log_param("learning_rate",0.01)

        print(f"Run ID:{myrun.info.run_id}")
        print(f"Experimrnt ID:{myrun.info.experiment_id}")
        print(f"Status:{myrun.info.status}")
        print(f"Start Time:{myrun.info.start_time}")
        print(f"End Time:{myrun.info.end_time}")
        print(f"Lifecycle Stage:{myrun.info.lifecycle_stage}")
