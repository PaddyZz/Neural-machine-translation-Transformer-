from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    model_name: str
    model_URL: str
    model_zip_file: Path
    model_unzip_dir: Path
    

@dataclass(frozen=True)
class PretrainModelConfig:
    root_dir: Path
    params_max_tokens: int
    params_buffer_size: int
    params_batch_size: int

@dataclass(frozen=True)
class TrainModelConfig:
    root_dir: Path
    params_max_epochs: int
    params_num_layers: int
    params_d_model: int
    params_dff: int
    params_num_heads: int
    params_dropout_rate: int
    params_beta_1: int
    params_beta_2: int
    params_epsilon: float

@dataclass(frozen=True)
class EvaluateConfig:
    root_dir: Path
    saved_json_dir: Path
    tf_model_saved_dir: Path
    keras_model_saved_dir: Path
    sentence_tbt: str
    ground_truth: str
    save_keras: bool