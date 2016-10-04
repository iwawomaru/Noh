import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh_.environments import LunarLander
from noh.components import PGNet

n_stat = LunarLander.n_stat
n_act = LunarLander.n_act
reward_reset_checker = LunarLander.gen_reward_reset_checker()
pg_net = PGNet(structure=[n_stat*4, 500, 500, 500, n_act],
               mbatch_size=10, lr=1e-6, epsilon=1.0, decay_rate=0.99, gamma=0.99,
               reward_reset_checker=reward_reset_checker, is_argmax=True)
noh_env = LunarLander(pg_net, render=True)

while True:
    noh_env.exec_episode()