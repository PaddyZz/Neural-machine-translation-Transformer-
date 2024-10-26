from src.transformer.constants import *
from src.transformer.utils.common import read_yaml, create_directories
from src.transformer.entity.config_entity import (DataIngestionConfig,
PretrainModelConfig,TrainModelConfig,EvaluateConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_name=config.dataset_name,
            model_name=config.model_name,
            model_unzip_dir=config.model_unzip_dir,
            model_URL=config.model_URL,
            model_zip_file=config.model_zip_file
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PretrainModelConfig:
        config = self.config.pretrain_model
        params = self.params
        create_directories([config.root_dir])

        pretrain_model_config = PretrainModelConfig(
            root_dir=Path(config.root_dir),
            params_max_tokens=params.MAX_TOKENS,
            params_buffer_size=params.BUFFER_SIZE,
            params_batch_size=params.BATCH_SIZE
        )

        return pretrain_model_config
    



    def get_training_config(self) -> TrainModelConfig:
        config = self.config.train_model
        params = self.params
        create_directories([
            Path(config.root_dir)
        ])

        train_model_config = TrainModelConfig(
            root_dir=Path(config.root_dir),
            params_max_epochs = params.MAX_EPOCHS,
            params_d_model= params.d_model,
            params_dff=params.dff,
            params_dropout_rate=params.dropout_rate,
            params_num_heads = params.num_heads,
            params_num_layers=params.num_layers,
            params_beta_1=params.beta_1,
            params_beta_2=params.beta_2,
            params_epsilon=params.epsilon
        )

        return train_model_config
    


    def get_evaluation_config(self) -> EvaluateConfig:
        config = self.config.evaluate_model
        params = self.params
        eval_config = EvaluateConfig(
            root_dir=Path(config.root_dir),
            saved_model_dir=Path(config.saved_model_dir),
            saved_json_dir=Path(config.saved_json_dir),
            tf_model_saved_dir = Path(config.tf_model_saved_dir),
            keras_model_saved_dir = Path(config.keras_model_saved_dir),
            save_keras = params.save_keras,
            sentence_tbt = params.sentence_tbt,
            ground_truth= params.ground_truth
        )
        return eval_config