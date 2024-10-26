import tensorflow as tf
from typing import Tuple
import tensorflow_datasets as tfds
from src.transformer.entity.config_entity import (DataIngestionConfig)


class data_ingestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def get_datesets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
            returns train and validation datasets based on pt-en translation datasets
        """
        try:
            examples, metadata = tfds.load(f'{self.config.dataset_name}',
                               with_info=True,
                               as_supervised=True)

            train_examples, val_examples = examples['train'], examples['validation']
            return train_examples, val_examples
        except Exception as e:
            raise e