import numpy as np

from noh.environment import UnsupervisedEnvironment

class SimpleUnsupervisedTest(UnsupervisedEnvironment):

    dataset = np.array([[1, 1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0]])

    test_dataset = np.array([[0, 0, 0, 1, 1, 0],
                             [1, 1, 0, 0, 0, 0]])

    def __init__(self, model):
        super(SimpleUnsupervisedTest, self).__init__(model)



