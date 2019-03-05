import sys

sys.path.insert(0, '..')

import argparse
from mxnet import gluon, init, nd, test_utils, cpu, gpu
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn
from mxnet.gluon.estimator import estimator as est
from utils import *


def corr1d(X, K):
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


def corr1d_multi_in(X, K):
    # First, we traverse along the 0th dimension (channel dimension) of X and
    # K. Then, we add them together by using * to turn the result list into a
    # positional argument of the add_n function
    return nd.add_n(*[corr1d(x, k) for x, k in zip(X, K)])


# Model
class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # The embedding layer does not participate in training
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no weight, so it can share an
        # instance
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate the output of two embedding layers with shape of
        # (batch size, number of words, word vector dimension) by word vector
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # According to the input format required by Conv1D, the word vector
        # dimension, that is, the channel dimension of the one-dimensional
        # convolutional layer, is transformed into the previous dimension
        embeddings = embeddings.transpose((0, 2, 1))
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, an NDArray with the shape of (batch size, channel size, 1)
        # can be obtained. Use the flatten function to remove the last
        # dimension and then concatenate on the channel dimension
        encoding = nd.concat(*[nd.flatten(
            self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # After applying the dropout method, use a fully connected layer to
        # obtain the output
        outputs = self.decoder(self.dropout(encoding))
        return outputs


if __name__ == '__main__':
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='MXNet Gluon Text Sentiment Classification Example using CNN')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training and testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='whether to use GPU (default: False)')
    opt = parser.parse_args()

    # ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    gpus = test_utils.list_gpus()
    ctx = [gpu(i) for i in gpus] if len(gpus) > 0 else [cpu()]

    # data
    download_imdb()
    train_data, test_data = read_imdb('train'), read_imdb('test')
    vocab = get_vocab_imdb(train_data)
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        *preprocess_imdb(train_data, vocab)), opt.batch_size, shuffle=True)
    test_iter = gdata.DataLoader(gdata.ArrayDataset(
        *preprocess_imdb(test_data, vocab)), opt.batch_size)

    # Initialize
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
    net.initialize(init.Xavier(), ctx=ctx)

    glove_embedding = text.embedding.create(
        'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.constant_embedding.collect_params().setattr('grad_req', 'null')

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr})
    loss = gloss.SoftmaxCrossEntropyLoss()

    e = est.Estimator(net=net, loss=loss, trainers=trainer, ctx=ctx)
    e.fit(train_iter, test_iter, opt.epochs, batch_size=opt.batch_size)
