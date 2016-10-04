import numpy as np

from noh.environment import UnsupervisedEnvironment

class SequenceBit(UnsupervisedEnvironment):

    n_visible = 100
    n_label = 0
    n_dataset = 10

    def __init__(self, model):
        super(SequenceBit, self).__init__(model)
        #self.dataset = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.dataset = []
        #for i in xrange(self.n_dataset):
        #    self.dataset.append(np.random.randint(0, 2, size=self.n_visible))
        self.dataset = [[1 if i==j else 0 for j in xrange(self.n_visible)] for i in xrange(self.n_dataset)]
        self.dataset = np.array(self.dataset)

    def train(self, epochs):
        super(SequenceBit, self).train(epochs=epochs)

    def unsupervised_train(self, epochs):
        self.model.unsupervised_train(data=self.dataset, epochs=epochs)

    def next_frame_estimation(self, step=1):
        for data in self.dataset:
            self.model(data)
            self.model(None)
            rec = self.model.prop_down()
            print rec

    def rec_check(self):
        res = self.model.rec_sequence(self.dataset)
        for line in res:
            for data in line[0]:
                print round(data, 1),
            print ""