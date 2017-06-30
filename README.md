# Sequence Tagging (Named Entity Recognition) with Tensorflow

This repo implements a sequence tagging model using tensorflow.

State-of-the-art performance (F1 score close to 91).

Check my [blog post](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)

## Task

Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
John lives in New York
PER  O     O  LOC LOC
```

## Model

Similar to [Lample et al.](https://arxiv.org/abs/1603.01360) and [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF

## Data

The training data must be in the following format (identical to the CoNLL2003 dataset).


```
John B-PER
lives O
in O
New B-LOC
York I-LOC
. O

This O
is O
another O
sentence
```


Once you have produced your txt files, change the parameters in `config.py` like

```
# dataset
dev_filename = "data/coNLL/eng/eng.testa.iob"
test_filename = "data/coNLL/eng/eng.testb.iob"
train_filename = "data/coNLL/eng/eng.train.iob"
```

You also need to download GloVe vectors.

## Usage

First, build vocab from the data and extract trimmed glove vectors according to the config in `config.py`.

```
python build_data.py
```

Second, train and test model with 

```
python main.py
```

Data iterators and utils are in `data_utils.py` and the model with training/test procedures are in `model.py`

Training time on NVidia Tesla K80 is 110 seconds per epoch on CoNLL train set using characters embeddings and CRF.


## License 

This project is licensed under the terms of the apache 2.0 license (as Tensorflow and derivatives). If used for research, citation would be appreciated.

