import wandb
from wandb.integration.keras import WandbMetricsLogger
import tensorflow as tf

from utils.models import MODELS
from utils.data import build_datasets

from utils.config import ExperimentConfig, WandbConfig

def build_model(config:ExperimentConfig , n_labels:int):
    model_fn = MODELS[config.model_name]

    model = model_fn(
        input_shape=config.input_shape,
        n_labels=n_labels
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=config.loss,
        metrics=["accuracy"]
    )

    return model

def run_experiment(X_train, y_train,
                   X_test, y_test,
                   wandb_config: WandbConfig,
                   config: ExperimentConfig,
                   class_names: list,
                   preprocess_fn=None):

    cfg = vars(config)

    run = wandb.init(
        project=wandb_config.project_name,
        config=cfg, # converts dataclass into dict
        name=wandb_config.experiment_name,
        tags=wandb_config.tags,  # Organize experiments
        notes=wandb_config.notes
    )

    # Build tf.data pipelines
    train_ds, val_ds, test_ds = build_datasets(
        X_train, y_train,
        X_test, y_test,
        config=config,
        preprocess_fn=preprocess_fn,
        val_split=config.val_split)

    n_labels = len(class_names)
    model = build_model(config, n_labels=n_labels)
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[WandbMetricsLogger(log_freq="epoch")]
    )

    # # Full evaluation on test set
    # log_per_class_metrics(model, test_ds, y_test, class_names, run)

    run.finish()