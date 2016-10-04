from abc import ABCMeta, abstractmethod
import warnings

class Component(object):
    """ Callable Component for Circuits with training capability.

    Bind, Circuit, and all component implementations defined in :mod:
    `noh_.components` inherit this class.

    :class: `Component`s are the basic building blocks for :class: `Circuits`
    which must provide interfaces to be called and trained. Every component is
    expected to have the :meth: `__call__` method return a meaningful value
    given some sort of input. This is because every :class: `Component` is
    treated as a function which calculates a mapping of a given input.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.params = {}

        self.input_hist = []
        self.output_hist = []
        self.error_hist = []
        self.reward_hist = []

    @abstractmethod
    def __call__(self, data, **kwargs):
        raise NotImplementedError("`__call__` must be explicitly overridden")

    def supervised_train(self, data, label, epochs, **kwargs):
        warnings.warn("supervised_train will do nothing")

    def unsupervised_train(self, data, label, epochs, **kwargs):
        warnings.warn("unsupervised_train will do nothing")

    def reinforcement_train(self, data, label, epochs, **kwargs):
        warnings.warn("reinforcemnt_train will do nothing")

    def save_params(self):
        raise NotImplementedError("`train` must be explicitly overridden")

