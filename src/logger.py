from pytorch_lightning.loggers import MLFlowLogger

def get_logger(experiment_name, run_name):
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name
    )