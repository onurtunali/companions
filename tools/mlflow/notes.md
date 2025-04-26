# MLFlow Notes

MLFlow is a machine learning model tracking tool with comprehensive API integration with popular ML libraries such as scikit-lern, pytorch, keras etc. MLFlow modules specific to different stakeholders are shown below.

![](https://mlflow.org/docs/latest/_images/mlflow-overview.png)

These are various notes and practices not obvious or included in documentation.

## MLFlow Deployment

Easiest way to deploy an MLFlow instance is locally. This can only be used for individual development. Since any other production level MLFlow would have a database and artifact store, it's good practice to mimic the same approach locally as well. Chage `<user_name>` with your own.

```sh
uv tool install mlflow
uv tool run mlflow server --backend-store-uri sqlite:////Users/<user_name>/platforms/mlflow/mlflow.db --default-artifact-root /Users/<user_name>/platforms/mlflow/mlflow_artifacts --host 0.0.0.0 --port 5000
```

Deleting expreiments or runs from UI does not clean artifacts and db records. For this purpose, MLFlow development team added a manual garbage collection approach. Run the following command for clean-up.

```sh
uv tool run mlflow gc --backend-store-uri sqlite:////Users/<user_name>/platforms/mlflow/mlflow.db --artifact-destination /Users/<user_name>/platforms/mlflow/mlflow_artifacts --host 0.0.0.0 --port 5000
```

## Best practices

MLflow `input_log` method logs the dataset information and metada however, it does NOT track the dataset itself.

```
# Ask MLFlow Answer:
{

    "prompt":" Does MLflow store the actual dataset when dataset is logged?",
    "answer": "It seems likely that MLflow does not store the actual dataset 
    when it is logged. Instead, MLflow logs metadata about the dataset, such 
    as its name, digest (hash), schema, and source. This metadata is used to 
    track the dataset's usage in experiments and runs. The actual data remains 
    in its original location, and MLflow records a reference to it. This approach 
    allows for efficient tracking without duplicating large datasets. Would you 
    like more detail on how MLflow handles dataset logging?"

}
```

Log system metrics functionality track the resource utilization of the whole host not just the training script.

```
# Ask MLFlow Answer:
{

    "prompt": "When system_metrics_enabled, does MLFlow tracks resource utilization of 
    the training script or the current metrics for the whole system? For example, let's 
    say host has two processes p1 and p2 such that p1 is system operations with 20% cpu 
    utilization and p2 is training script with 50% cpu utilization. Will cpu metrics be 
    70% indicating the whole cpu utilization or 50% just for the training process?",
    "answer": "MLflow tracks system-wide resource utilization metrics, not specific to the 
    training script. Therefore, in your example, the CPU metrics would indicate 70% utilization, 
    reflecting the total CPU usage of the host system, including both processes p1 and p2. 
    Would you like more details on how MLflow logs system metrics?

    source https://github.com/mlflow/mlflow/issues/12916"
}
```

To track `numpy` based datasets with `mlflow.data.from_numpy`, features array should have `flatten` method for infering dataset shape. For this reason, `scipy` based `csr` matrix data generally generated from sparse feature extraction methods cannot be used. If data can be load into memory, `x.toarray()` can be passed to method, however if dense version of data causes out of memory erro, then it's better to track dataset with a custom tag functionality.

For hyperparameter sweeps, use child runs otherwise expreiment namespace get cluttered pretty quickly.

```python
import mlflow
import model

with mlflow.start_run() as parent_run:
    params = [{"param1": v1, "param2": v2}, ...]

    for p in params:
        with mlflow.start_run(nested=True) as child_run:
            model.set_params(p)
            mlflow.log_params(p)
```

Definiton of "artifact" is very context dependent regarding machine learning. However, any static file(s) or directory can be tracked with `mlflow.log_artifacs` method.