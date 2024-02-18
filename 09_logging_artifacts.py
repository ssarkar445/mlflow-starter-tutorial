import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")

    print(f"Name:{experiment.experiment_id}")

    with mlflow.start_run(run_name="logging_artifacts",experiment_id=experiment.experiment_id) as myrun:

        # your machine learning code goes here

        with open("hello_world2.txt","w") as f:
            f.write("Hello World!")

        mlflow.log_artifact(local_path="hello_world2.txt",artifact_path="test_file")

        print(f"Run ID:{myrun.info.run_id}")
        print(f"Experimrnt ID:{myrun.info.experiment_id}")
        print(f"Status:{myrun.info.status}")
        print(f"Start Time:{myrun.info.start_time}")
        print(f"End Time:{myrun.info.end_time}")
        print(f"Lifecycle Stage:{myrun.info.lifecycle_stage}")

