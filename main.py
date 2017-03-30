import os
from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset
from general_utils import get_logger
from model import NERModel
from config import config

if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

vocab_words = load_vocab(config.words_filename)
vocab_tags  = load_vocab(config.tags_filename)
vocab_chars = load_vocab(config.chars_filename)

processing_word = get_processing_word(vocab_words, vocab_chars, 
    lowercase=config.lowercase, chars=config.chars)
processing_tag  = get_processing_word(vocab_tags, lowercase=False)

embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

dev = CoNLLDataset(config.dev_filename, processing_word, 
    processing_tag, config.max_iter)
test = CoNLLDataset(config.test_filename, processing_word, 
    processing_tag, config.max_iter)
train = CoNLLDataset(config.train_filename, processing_word, 
    processing_tag, config.max_iter)

logger = get_logger(config.log_path)
model = NERModel(config, embeddings, nchars=len(vocab_chars), logger=logger)
model.build()
model.train(train, dev, vocab_tags)
model.evaluate(test, vocab_tags)
model.interactive_shell(vocab_tags, processing_word)