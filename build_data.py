from config import config
from data_utils import CoNLLDataset, get_vocab, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, export_trimmed_glove_vectors


def build_data(config):
    """
    Procedure to bulid data
    Args:
        config: defines attributes needed in the function
    Return:
        creates a vocab file
        creates a npz embedding file
    """
    dev   = CoNLLDataset(config.dev_filename)
    test  = CoNLLDataset(config.test_filename)
    train = CoNLLDataset(config.train_filename)

    vocab_words, vocab_tags = get_vocab([dev], config.lowercase)
    vocab_glove = get_glove_vocab(config.glove_filename)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    write_vocab(vocab, config.vocab_filename)
    write_vocab(vocab_tags, config.tags_filename)

    vocab = load_vocab(config.vocab_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename, 
                                config.trimmed_filename, config.dim)


if __name__ == "__main__":
    build_data(config)