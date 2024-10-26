# Neural machine translation(Transformer)

## Introduction
Neural Machine Translation (NMT) has revolutionized the field of language translation by leveraging the power of deep learning models. Among various architectures, the Transformer model has emerged as a groundbreaking approach, significantly improving translation quality and efficiency. This repository provides a comprehensive implementation of a Transformer-based NMT system using Keras, an accessible and powerful deep learning framework.

The Transformer architecture, introduced by Vaswani et al. in the seminal paper "Attention is All You Need," departs from traditional sequence-to-sequence models by eliminating recurrent layers in favor of self-attention mechanisms. This allows the model to capture long-range dependencies more effectively and facilitates parallelization during training, resulting in faster convergence and improved performance.

In this repository, a code implementing based on a basic transformer architecture to building an NMT system is presented, starting from build tokenizer to model training and evaluation. A pretrain model and  Scripts are also provided for setting up tokenizer and translating new and personalized text from Portuguese to English, enabling users to quickly and experiment with the system. 

## Get Started
```python
# create a new conda env and install the dependency file
conda create --name <env-name> python=3.10.3
conda activate <env-name>
pip install -r requirements.txt
```

## Execute
```python
python main.py
#or
python main.py [-c | --config] <params=value>
```

## Optional Parameters

- **`MAX_EPOCHS`**: Set the maximum number of training epochs (e.g., `20`), type is integer, default is `20`.

- **`MAX_TOKENS`**: Set the maximum number of tokens in each input sequence (e.g., `128`), type is integer, default is `128`.

- **`BUFFER_SIZE`**: Set the size of the buffer for shuffling the dataset (e.g., `20000`), type is integer, default is `20000`.

- **`BATCH_SIZE`**: Set the number of samples per gradient update (e.g., `64`), type is integer, default is `64`.

- **`num_layers`**: Set the number of layers in the encoder and decoder (e.g., `4`), type is integer, default is `4`.

- **`d_model`**: Set the dimensionality of the model's output embeddings (e.g., `128`), type is integer, default is `128`.

- **`dff`**: Set the dimensionality of the feedforward layer (e.g., `512`), type is integer, default is `512`.

- **`num_heads`**: Set the number of attention heads (e.g., `8`), type is integer, default is `8`.

- **`dropout_rate`**: Set the dropout rate to prevent overfitting (e.g., `0.1`), type is float, default is `0.1`.

- **`beta_1`**: Set the exponential decay rate for the first moment estimates in the Adam optimizer (e.g., `0.9`), type is float, default is `0.9`.

- **`beta_2`**: Set the exponential decay rate for the second moment estimates in the Adam optimizer (e.g., `0.98`), type is float, default is `0.98`.

- **`epsilon`**: Set a small constant for numerical stability in the Adam optimizer (e.g., `1e-9`), type is float, default is `1e-9`.

- **`sentence_tbt`**: Set the input sentence to be translated (e.g., `'este é um problema que temos que resolver.'`), type is string, default is `''`.

- **`ground_truth`**: Set the expected output used to compared with the translaton text (e.g., `'this is a problem we have to solve.'`), type is string, default is `''`.

- **`save_keras`**: Set a flag to check if save the Keras model after training (e.g., `false`), type is boolean, default is `false`. default saving way is tf.saved_model.save()


## Personalized  the tokenizer and datasets 

```python
# Personalized the loading ways of the tokenizer and datasets for translate other languages to English
# /root_dir/config/config.yaml
# assume your pretrained model dir is like 'https://storage.googleapis.com/download.tensorflow.org/models/ted_hrlr_translate_pt_en_converter.zip
dataset_name: <replace you own dataset name like 'ted_hrlr_translate/pt_to_en' to allow tfds loading>
model_name: <replace your own pretrain model name like 'ted_hrlr_translate_pt_en_converter'>
model_URL: <replace your own url like 'https://storage.googleapis.com/download.tensorflow.org/models/'>
```

## Dockerfile

```
FROM python:3.10.3
RUN pip install virtualenv
RUN virtualenv /env
ENV VIRTUAL_ENV=/env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
```

```bash
docker build
```

## Blog
blog_link

## Reference
[build a basic transformer architecture](https://www.tensorflow.org/text/tutorials/transformer)
