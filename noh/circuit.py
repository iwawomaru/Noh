from abc import ABCMeta, abstractmethod

from noh.component import Component
from noh.utils import DotAccessible

class PropRule(object):

    __metaclass__ = ABCMeta

    def __init__(self, components):
        self.components = DotAccessible(components)

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class TrainRule(object):

    __metaclass__ = ABCMeta

    def __init__(self, components):
        self.components = DotAccessible(components)

    @abstractmethod
    def __call__(self, data, label, epoch):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class Planner(object):
    def __init__(self, components, prop, train, **Rules):
        self.components = DotAccessible(components)
        self.rules = {
            'prop': prop(components),
            'train': train(components)
        }

        for name in Rules:
            Rule = Rules[name]
            self.rules[name] = Rule(components)

        self.prop_rule = self.rules['prop']
        self.train_rule = self.rules['train']

    def set_prop(self, name):
        self.prop_rule = self.rules[name]

    def set_train(self, name):
        self.train_rule = self.rules[name]

    def __call__(self, data):
        return self.prop_rule(data)

    def train(self, data, label, epoch):
        return self.train_rule(data, label, epoch)

class Circuit(Component):
    def __init__(self, planner, **components):
        self.components = DotAccessible(components)
        self.planner = planner(components)

    def __call__(self, data):
        return self.planner(data)

    def train(self, data, label, epochs):
        return self.planner.train(data, label, epochs)
