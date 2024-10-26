import tensorflow as tf
from src.transformer.entity.config_entity import (PretrainModelConfig, DataIngestionConfig)
import tensorflow_text
from src.transformer.config.configuration import (ConfigurationManager)

class pretrain_model:    
    def __init__(self, config: PretrainModelConfig, dataIngestionConfig: DataIngestionConfig):
        self.config = config
        self.dataIngestionConfig = dataIngestionConfig

    def get_tokenizer(self) -> tf.keras.Model:

        try:
            
            tf.keras.utils.get_file(
                f'{self.dataIngestionConfig.model_name}.zip',
                f'{self.dataIngestionConfig.model_URL}/{self.dataIngestionConfig.model_name}.zip',
                cache_dir='.', cache_subdir='', extract=True
            )
            tokenizers = tf.saved_model.load(self.dataIngestionConfig.model_name)
            return tokenizers
        except Exception as e:
            raise e
    
def prepare_batch(pt, en):
    tokenizers = pretrain_model().get_tokenizer()
    pre_model_config = ConfigurationManager().get_prepare_base_model_config()
    MAX_TOKENS = pre_model_config.params_max_tokens
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels

def make_batches(ds):
    pre_model_config = ConfigurationManager().get_prepare_base_model_config()
    return (
        ds
        .shuffle(pre_model_config.params_buffer_size)
        .batch(pre_model_config.params_batch_size)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    