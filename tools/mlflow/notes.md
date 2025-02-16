# MLFlow Notes

MLFlow is a machine learning model tracking tool with comprehensive API integration with popular ML libraries such as scikit-lern, pytorch, keras etc. These are various notes and practices not obvious or included in documentation.

MLflow `input_log` method logs the dataset information and metada however, it does NOT track the dataset itself.

```json
// Ask MLFlow Answer:
{
"prompt":" Does MLflow store the actual dataset when dataset is logged?",
"Answer": "It seems likely that MLflow does not store the actual dataset when it is logged. Instead, MLflow logs metadata about the dataset, such as its name, digest (hash), schema, and source. This metadata is used to track the dataset's usage in experiments and runs. The actual data remains in its original location, and MLflow records a reference to it. This approach allows for efficient tracking without duplicating large datasets. Would you like more detail on how MLflow handles dataset logging?"
}
```

Log system metrics functionality track the resource utilization of the whole host not just the training script.

```json
// Ask MLFlow Answer:
{
"prompt": "When system_metrics_enabled, does MLFlow tracks resource utilization of the training script or the current metrics for the whole system? For example, let's say host has two processes p1 and p2 such that p1 is system operations with 20% cpu utilization and p2 is training script with 50% cpu utilization. Will cpu metrics be 70% indicating the whole cpu utilization or 50% just for the training process?",
"answer": "MLflow tracks system-wide resource utilization metrics, not specific to the training script. Therefore, in your example, the CPU metrics would indicate 70% utilization, reflecting the total CPU usage of the host system, including both processes p1 and p2. Would you like more details on how MLflow logs system metrics?

sources https://github.com/mlflow/mlflow/issues/12916"
}
```

To track `numpy` based datasets with `mlflow.data.from_numpy`, features array should have `flatten` method for infering dataset shape. For this reason, `scipy` based `csr` matrix data generally generated from sparse feature extraction methods cannot be used. If data can be load into memory, `x.toarray()` can be passed to method, however if dense version of data causes out of memory erro, then it's better to track dataset with a custom tag functionality.
