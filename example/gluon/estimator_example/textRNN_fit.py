import sys

sys.path.insert(0, '..')

import argparse
import collections
import d2l
import mxnet as mx
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
from mxnet.gluon.estimator import estimator as est
import os
import random
import tarfile


# Download data
def download_imdb(data_dir='../data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)


def read_imdb(folder='train'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


def get_tokenized_imdb(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)


def preprocess_imdb(data, vocab):
    # Make the length of each comment 500 by truncating or adding 0s
    max_l = 500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels


# Model
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # Set Bidirectional to True to get a bidirectional recurrent neural
        # network
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of inputs is (batch size, number of words). Because LSTM
        # needs to use sequence as the first dimension, the input is
        # transformed and the word feature is then extracted. The output shape
        # is (number of words, batch size, word vector dimension).
        embeddings = self.embedding(inputs.T)
        # The shape of states is (number of words, batch size, 2 * number of
        # hidden units).
        states = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * number of hidden units)
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs


if __name__ == '__main__':

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='MXNet Gluon Text Sentiment Classification Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training and testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='whether to use GPU (default: False)')
    opt = parser.parse_args()

    ctx = d2l.try_all_gpus() if opt.use_gpu else [mx.cpu()]

    # data
    download_imdb()
    train_data, test_data = read_imdb('train'), read_imdb('test')
    vocab = get_vocab_imdb(train_data)
    '# Words in vocab:', len(vocab)

    train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
    test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
    train_iter = gdata.DataLoader(train_set, opt.batch_size, shuffle=True)
    test_iter = gdata.DataLoader(test_set, opt.batch_size)

    for X, y in train_iter:
        print('X', X.shape, 'y', y.shape)
        break
    '#batches:', len(train_iter)

    embed_size, num_hiddens, num_layers = 100, 100, 2

    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    net.initialize(init.Xavier(), ctx=ctx)

    glove_embedding = text.embedding.create(
        'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)

    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.embedding.collect_params().setattr('grad_req', 'null')

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr})
    loss = gloss.SoftmaxCrossEntropyLoss()

    e = est.Estimator(net, trainer, loss=loss, ctx=ctx)
    e.fit(train_iter, test_iter, opt.epochs, batch_size=opt.batch_size)
