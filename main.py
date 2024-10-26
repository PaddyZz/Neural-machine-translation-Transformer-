from src.transformer import logger

from src.transformer.config.configuration import (ConfigurationManager)
from src.transformer.components.pretrain_model import (pretrain_model, make_batches)
from src.transformer.components.train_model import (Transformer, CustomSchedule, masked_accuracy,masked_loss)
from src.transformer.components.Translator import *
from src.transformer.components.data_ingestion import data_ingestion
import tensorflow as tf 

STAGE_NAME ="DATA_INGESTION"
STAGE_NAME_ONE = "PRETRAIN_MODEL"
STAGE_NAME_TWO = "TRAINING_MODEL"
STAGE_NAME_THREE = "TRANSLATE_TEXT"

try:
 
    PMC = ConfigurationManager().get_prepare_base_model_config()
    DIC = ConfigurationManager().get_data_ingestion_config()
    TC  = ConfigurationManager().get_training_config()
    eval_config = ConfigurationManager().get_evaluation_config()
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    pretrain_model_example = pretrain_model(PMC, DIC)
    tokenizers = pretrain_model_example.get_tokenizer()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    logger.info(f">>>>>> stage {STAGE_NAME_ONE} started <<<<<<")
    transformer = Transformer(
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),)


    learning_rate = CustomSchedule(TC.params_d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=TC.params_beta_1, beta_2=TC.params_beta_2,
                                        epsilon=float(TC.params_epsilon))

    transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])
    

    train_examples, val_examples = data_ingestion(DIC).get_datasets()
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)
    logger.info(f">>>>>> stage {STAGE_NAME_ONE} completed <<<<<<\n\nx==========x")
    logger.info(f">>>>>> stage {STAGE_NAME_TWO} started <<<<<<")
    transformer.fit(train_batches,
                epochs=TC.params_max_epochs,
                validation_data=val_batches)

    translator = Translator(tokenizers,transformer)
    sentence = eval_config.sentence_tbt
    ground_truth = eval_config.ground_truth

    translated_text, translated_tokens, attention_weights = translator(
        tf.constant(sentence))
    logger.info(f">>>>>> stage {STAGE_NAME_TWO} completed <<<<<<\n\nx==========x")
    logger.info(f">>>>>> stage {STAGE_NAME_THREE} started <<<<<<")
    save_model(transformer,config=eval_config)
    print_translation(sentence, translated_text, ground_truth)

    save_into_json(translated_text, config=eval_config)
    logger.info(f">>>>>> stage {STAGE_NAME_THREE} completed <<<<<<\n\nx==========x")

except Exception as e:
    raise e