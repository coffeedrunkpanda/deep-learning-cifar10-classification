from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    dataset: str = "CIFAR-10"
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 32
    model_name: str = "baseline_cnn"
    loss:str = "sparse_categorical_crossentropy"
    val_split:float = 0.2
    input_shape:tuple = (32, 32, 3)
    normalize:bool = True

@dataclass
class WandbConfig:
    project_name: str
    experiment_name:str
    tags: list = field(default_factory=lambda: ["baseline", "cnn", "scratch"])
    notes:str = "Simple 3-layer CNN with batch normalization"
