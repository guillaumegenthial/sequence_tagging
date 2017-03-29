import numpy as np
import os

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

class CoNLLDataset(object):
    """
    Class that iterates over CoNLL Dataset
    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None, 
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```
    """
    def __init__(self, filename, processing_word=None, processing_tag=None, 
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    word, tag = line.split(' ')
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocab(datasets, lowercase=False):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print "Building vocab..."
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            if lowercase:
                words = map(lambda x: x.lower(), words)
            vocab_words.update(words)
            vocab_tags.update(tags)
    print "- done. {} words".format(len(vocab_words))
    return vocab_words, vocab_tags


def get_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print "Building vocab..."
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print "- done. {} words".format(len(vocab))
    return vocab


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file
    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Return:
        write a word per line
    """
    print "Writing vocab..."
    with open(filename, "w") as f:
        for word in vocab:
            f.write("{}\n".format(word))
    print "- done. {} words".format(len(vocab))


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx

    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = map(float, line[1:])
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    filename: path to the npz file
    """
    with open(filename) as f:
        return np.load(f)["embeddings"]


def get_vocab_processing(vocab, lowercase=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns: 
        function word -> idx
    """
    def f(word):
        if lowercase:
            word = word.lower()
        if word.isdigit():
            return vocab[NUM]
        if word in vocab:
            return vocab[word]
        else:
            return vocab[UNK]

    return f


def pad_sequences(sequences, pad_tok):
    sequence_padded, sequence_length = [], []
    max_length = max(map(lambda x : len(x), sequences))

    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('-')[-1]


def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.iteritems()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks