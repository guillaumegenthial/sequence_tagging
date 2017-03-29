import os
from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_vocab_processing, CoNLLDataset
from general_utils import get_logger
from model import NERModel
from config import config

if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

embeddings = get_trimmed_glove_vectors(config.trimmed_filename)
vocab_words = load_vocab(config.vocab_filename)
processing_word = get_vocab_processing(vocab_words, config.lowercase)
vocab_tags = load_vocab(config.tags_filename)
processing_tag = get_vocab_processing(vocab_tags, False)

dev = CoNLLDataset(config.dev_filename, processing_word, 
    processing_tag, config.max_iter)
test = CoNLLDataset(config.test_filename, processing_word, 
    processing_tag, config.max_iter)
train = CoNLLDataset(config.train_filename, processing_word, 
    processing_tag, config.max_iter)

logger = get_logger(config.log_path)
model = NERModel(config, embeddings, logger)
model.build()
model.train(train, dev, vocab_tags)
model.evaluate(test, vocab_tags)
model.interactive_shell(vocab_tags, processing_word)