import mlflow
from mlflow_utils import get_mlflow_experiment


if __name__=="__main__":

    # retrieve the mlflow experiment
    experiment = get_mlflow_experiment(experiment_id='833549759918707970')

    print(f"Name: {experiment.name}")
    print(f"Experimrnt ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")
