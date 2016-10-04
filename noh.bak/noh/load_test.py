class Resv(Component):
    def __init__(self, load_name=None):
        if load_name is None:
            self = pickle.load(load_name)
            return


class ESN(Circuit):
    def __init__(self, ...):
        super(ESN, self).__init__(
            planner=planner,
            components=[Resv(n_vis, n_hid, load_name="resv_hoge")])
    
