import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import wandb
from wandb.integration.keras import WandbMetricsLogger

from utils.data import build_datasets
from utils.models import MODELS
from utils.config import ExperimentConfig, WandbConfig

def build_model(config:ExperimentConfig , n_labels:int):
    model_fn = MODELS[config.model_name]

    model = model_fn(
        config=config,
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
    

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[
            WandbMetricsLogger(log_freq="epoch"),  # Auto-logs all metrics
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    models_dir = "models/"

    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)


    model.save("models/best_model.keras")
    wandb.log_artifact("models/best_model.keras", type="model")

    # Evaluate and log results: 
    val_loss, val_accuracy = model.evaluate(test_ds)
    wandb.log({
        "final_val_loss": val_loss,
        "final_val_accuracy": val_accuracy
    })

    # log images to wandb

    image_batch, label_batch = next(iter(test_ds))
    y_pred_probs = model.predict(image_batch)
    y_pred = np.argmax(y_pred_probs, axis=1)

    wandb_images = [
        wandb.Image(
            image_batch[i].numpy(),
            caption=f"True: {class_names[label_batch[i].numpy().flatten()[0]]} | Pred: {class_names[y_pred[i]]}"
        )
        for i in range(len(image_batch))  
    ]
    run.log({"prediction_samples": wandb_images})


    run.finish()


    return train_ds, val_ds, test_ds, model, history, run

    # TODO: add f1, precision, recall, auc
    