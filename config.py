class config():
    dim = 50
    glove_filename = "data/glove.6B/glove.6B.{}d.txt".format(dim)
    trimmed_filename = "data/glove.6B.{}d.trimmed.npz".format(dim)
    vocab_filename = "data/vocab.txt"
    tags_filename = "data/tags.txt"
    dev_filename = "data/coNLL/eng/eng.testa.iob"
    test_filename = "data/coNLL/eng/eng.testb.iob"
    train_filename = "data/coNLL/eng/eng.train.iob"
    max_iter = None
    lowercase = True
    train_embeddings = False
    ntags = 9
    nepochs = 5
    dropout = 0.5
    batch_size = 20
    lr = 0.001
    lr_decay = 0.9

    hidden_size = dim
    crf = False # if crf, training is 1.7x slower
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
