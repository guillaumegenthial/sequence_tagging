import numpy as np
import os
import tensorflow as tf
from data_utils import minibatches, pad_sequences, get_chunks
from general_utils import Progbar, print_sentence


class NERModel(object):
    def __init__(self, config, embeddings, logger=None):
        self.config = config
        self.embeddings = embeddings
        
        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        self.logger = logger


    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], 
                        name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], 
                        name="dropout")

        self.lr = tf.placeholder(dtype=tf.float32, shape=[], 
                        name="lr")


    def get_feed_dict(self, word_ids, sequence_lengths, labels=None, 
                      lr=None, dropout=None):
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }
        if labels is not None:
            feed[self.labels] = labels
        if lr is not None:
            feed[self.lr] = lr
        if dropout is not None:
            feed[self.dropout] = dropout

        return feed


    def add_word_embeddings_op(self):
        embeddings = tf.Variable(self.embeddings, name="embeddings", dtype=tf.float32, 
                                trainable=self.config.train_embeddings)

        self.word_embeddings = tf.nn.embedding_lookup(embeddings, self.word_ids, 
            name="word_embeddings")


    def add_logits_op(self):
        
        with tf.variable_scope("bi-lstm"):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, 
                lstm_cell, self.word_embeddings, sequence_length=self.sequence_lengths, 
                dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2*self.config.hidden_size, self.config.ntags], 
                dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.config.ntags], dtype=tf.float32, 
                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.config.ntags])

    def add_pred_op(self):
        if not self.config.crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        """
        Defines self.loss
        """
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)


    def add_train_op(self):
        """
        Defines self.train_op
        """
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)


    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess): 
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)


    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()


    def predict_batch(self, sess, word_ids, sequence_lengths):
        fd = self.get_feed_dict(word_ids, sequence_lengths, dropout=1.0)

        if self.config.crf:
            viterbi_sequences = []
            logits, transition_params = sess.run([self.logits, self.transition_params], 
                    feed_dict=fd)
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                logit, transition_params)
                viterbi_sequences += [viterbi_sequence]

            return viterbi_sequences

        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred


    def run_epoch(self, sess, train, dev, tags, epoch):
        nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (word_ids, labels) in enumerate(minibatches(train, self.config.batch_size)):
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            labels, _ = pad_sequences(labels, 0)

            fd = self.get_feed_dict(word_ids, sequence_lengths, labels, 
                                    self.config.lr, self.config.dropout)

            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        acc, f1 = self.run_evaluate(sess, dev, tags)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100*acc, 100*f1))
        return acc, f1


    def run_evaluate(self, sess, test, tags):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for word_ids, labels in minibatches(test, self.config.batch_size):
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)

            labels_pred = self.predict_batch(sess, word_ids, sequence_lengths)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += map(lambda (a, b): a == b, zip(lab, lab_pred))

                lab_chunks = set(get_chunks(lab, tags))
                lab_pred_chunks = set(get_chunks(lab_pred, tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1


    def train(self, train, dev, tags):
        best_score = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
                acc, f1 = self.run_epoch(sess, train, dev, tags, epoch)
                self.config.lr *= self.config.lr_decay
                if acc >= best_score:
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = acc
                    self.logger.info("- new best score!")


    def evaluate(self, test, tags):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.config.model_output)
            acc, f1 = self.run_evaluate(sess, test, tags)
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(100*acc, 100*f1))


    def interactive_shell(self, tags, processing_word):
        idx_to_tag = {idx: tag for tag, idx in tags.iteritems()}
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.config.model_output)
            self.logger.info("This is an interactive mode, enter a sentence:")
            while True:
                # Create simple REPL
                try:
                    sentence = raw_input("input> ")
                    words = sentence.strip().split(" ")
                    word_ids = map(processing_word, words)
                    print word_ids
                    preds_ids = self.predict_batch(sess, [word_ids], [len(word_ids)])[0]
                    preds = map(lambda idx: idx_to_tag[idx], list(preds_ids))
                    print_sentence(self.logger, {"x": words, "y": preds})
                except EOFError:
                    print("Closing session.")
                    break


