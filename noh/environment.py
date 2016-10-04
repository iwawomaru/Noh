from abc import ABCMeta, abstractmethod


class Environment(object):

    __metaclass__ = ABCMeta

    def __init__(self, model, render=False):
        self.model = model

        self.render = render
        self.bookmark = 0

        self.dataset = None
        self.test_dataset = None

    @classmethod
    def get_dataset(cls):
        return cls.dataset

    @classmethod
    def get_test_dataset(cls):
        return cls.test_dataset


    def supervised_train(self):
        self.model.supervised_train()

    def unsupervised_train(self):
        self.model.unsupervised_train()

    def reinforcement_train(self):
        self.model.reinforcement_train()

    def miso_soup(self, epochs=None):
        pass

    def reset(self):
        pass

    def print_stat(self):
        pass


