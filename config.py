class config():
    dim = 300
    dim_char = 100
    glove_filename = "data/glove.6B/glove.6B.{}d.txt".format(dim)
    trimmed_filename = "data/glove.6B.{}d.trimmed.npz".format(dim)
    words_filename = "data/words.txt"
    tags_filename = "data/tags.txt"
    chars_filename = "data/chars.txt"
    dev_filename = "data/coNLL/eng/eng.testa.iob"
    test_filename = "data/coNLL/eng/eng.testb.iob"
    train_filename = "data/coNLL/eng/eng.train.iob"
    max_iter = None
    lowercase = True
    train_embeddings = False
    nepochs = 20
    dropout = 0.5
    batch_size = 20
    lr = 0.001
    lr_decay = 0.9
    nepoch_no_imprv = 3

    hidden_size = 300
    char_hidden_size = 100
    crf = True # if crf, training is 1.7x slower
    chars = True # if char embedding, training is 3.5x slower
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
