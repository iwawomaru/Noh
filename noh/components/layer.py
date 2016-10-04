import numpy as np
from noh.activate_functions import sigmoid, p_sig
from noh.training_functions import gen_sgd_trainer

from noh.component import Component


class Layer(Component):
    def __init__(self, n_visible, n_hidden, activate=sigmoid):
        super(Layer, self).__init__()
        a = 1. / n_visible

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.array(np.random.uniform(low=-a, high=a, size=(n_visible, n_hidden)), dtype=np.float32)
        self.b_visible = np.zeros(n_visible, dtype=np.float32)
        self.b_hidden = np.zeros(n_hidden, dtype=np.float32)

        self.lr = 0.01

        self.activate = activate
        self.rng = np.random.RandomState(123)

    def __call__(self, data, label=None, reward=None, **kwargs):

        t = self.prop_up(data)
        error = 0.
        if label is not None:
            error = t - label
            self.error_hist.append(error)
        else:
            self.error_hist.append(None)

        self.input_hist.append(data)
        self.output_hist.append(t)
        self.reward_hist.append(reward)

        return sum(error)**2

    def supervised_train(self, epochs=1):
        e = np.array(self.error_hist)[0]
        x = np.array(self.input_hist)[0]
        y = np.array(self.output_hist)[0]
        for _ in xrange(epochs):
                self.W += self.lr * np.dot((p_sig(y) * e).T, x).T
                self.b_hidden += self.lr * np.dot((p_sig(y) * e).T, np.ones((self.n_hidden, 1)))[0]

        input_error = np.dot(e, self.W.T)
        return input_error

    def prop_up(self, v):
        return self.activate(np.dot(v, self.W) + self.b_hidden)

    def prop_down(self, h):
        return self.activate(np.dot(h, self.W.T) + self.b_visible)

    def rec(self, v):
        return self.prop_down(self.prop_up(v))

    def get_rec_square_error(self, v):
        rec_v = self.rec(v)
        return 0.5 * np.sum((v - rec_v)**2) / v.shape[0]

    def get_rec_cross_entropy(self, v):
        rec_v = self.rec(v)
        return - np.mean(np.sum(v * np.log(rec_v) + (1 - v) * np.log(1 - rec_v), axis=1))
