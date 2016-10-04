import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import MNISTEnv
from noh.components import Layer

if __name__ == "__main__":

    model = Layer(n_visible=28*28, n_hidden=10)
    env = MNISTEnv(model)

    env.miso_soup(n=10)
    env.supervised_train()
