import numpy as np
import warnings
from noh.components import RBM
from noh_.utils import get_lr_func
from noh_.activate_functions import sigmoid, p_sig

class PtReservoir(RBM):
    def __init__(self, n_visible, n_hidden, lr_type="hinton_r_div", r_div=None, lr=None):
        super(PtReservoir, self).__init__(n_visible=(n_visible+n_hidden), n_hidden=n_hidden, 
                                          lr_type=lr_type, r_div=r_div, lr=lr)
        self.prev_hid = np.zeros((1, n_hidden)) + 0.5

    def __call__(self, data):
        return self.prop_up(data)

    def train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        return self.unsupervised_train(data,  k=k, epochs=epochs)

    def prop_up(self, data):
        if data is None:
            data = np.zeros(shape=(1, self.n_visible-self.n_hidden)) + 0.5
        data = np.atleast_2d(data)
        data = np.c_[data, self.prev_hid]
        self.prev_hid = super(PtReservoir, self).prop_up(data)
        return self.prev_hid

    def prop_up_sequence(self, dataset):
        hid_list = []
        for data in dataset:
            hid_list.append(self.prop_up(data))
        return np.array(hid_list)

    def prop_down(self, data=None):
        if data is None:
            data = self.prev_hid
        data = super(PtReservoir, self).prop_down(h=data)
        data, hid = np.split(data, [self.n_visible-self.n_hidden], axis=1)
        return data

    def supervised_train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        super(RBM, self).train(data, label=None, lr=lr, k=k, epochs=epochs)

    def unsupervised_train(self, data, epochs=1000, minibatch_size=1, k=1):
        if minibatch_size != 1:
            warnings.warn("minibatch_size is not 1")
        for i in xrange(epochs):
            train_dataset = []
            for d in data:
                d = np.atleast_2d(d)
                joint_data = np.c_[d, self.prev_hid]
                self.prop_up(d)
                self.prev_hid = self.rng.binomial(size=self.prev_hid.shape, n=1, p=self.prev_hid)
                train_dataset.append(joint_data[0])
            train_dataset = np.array(train_dataset)
            print train_dataset
            self.CD(data=train_dataset)
            error = 0
            for d in train_dataset:
                error += super(PtReservoir, self).get_rec_cross_entropy(d)
            print "epoch: ", i, error / len(data)


    def get_rec_cross_entropy(self, v):
        v = np.atleast_2d(v)
        joint_v = np.c_[v, self.prev_hid]
        h = super(PtReservoir, self).prop_up(joint_v)
        rec_v = super(PtReservoir, self).prop_down(h)
        return - np.mean(np.sum(joint_v * np.log(rec_v) + (1 - joint_v) * np.log(1 - rec_v), axis=1))

    #def rec(self, v):
    #    raise NotImplementedError("rec is not implemented now.")

    def rec_sequence(self, dataset):
        res = []
        for data in dataset:
            h = self.prop_up(data)
            h = self.rng.binomial(n=1, p=h, size=h.shape)
            v = self.prop_down(h)
            res.append(v)
        return np.array(res)

    def get_energy(self, data):
        raise NotImplementedError("get_energy is not implemented now.")

    def gen_sampled_data(self, hidden_rep=None, n_sample=1, sampling_epochs=1):
        raise NotImplementedError("gen_sampled_data is not implemented now.")
