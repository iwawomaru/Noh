import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.components import PtReservoir
from noh_.environments import SequenceBit
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n_visible = SequenceBit.n_visible
    n_hidden = 100
    #pt_resv = PtReservoir(n_visible=n_visible, n_hidden=n_hidden, lr_type="hinton_r_div", r_div=1000)
    pt_resv = PtReservoir(n_visible, n_hidden, lr_type="const", lr=0.01)
    env = SequenceBit(pt_resv)

    env.unsupervised_train(epochs=1000)
    env.rec_check()
    print "-------"
    env.rec_check()

    #env.next_frame_estimation()


