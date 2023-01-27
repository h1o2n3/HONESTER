import os 


class Dataset():
    def __init__(self):    
        self.wikipedia = 'wikipedia'
        self.reddit = 'reddit'
        self.math = 'mathoverflow-a2q'
        self.bitcoinalpha = 'bitcoinalpha'
        self.bitcoinotc = 'bitcoinotc'

PATH = os.path.join(os.getcwd())


dataset = Dataset()




PARAMETER_DICT = {
    'log_name':'honester',
    'data_name':dataset.bitcoinalpha,
    'batch':512,
    'n_epoch':15,
    'mode' : 'i',
    'n_head':3,
    'drop_out':0.1,
    'attn_mode':'prod',
    'n_layer':2,
    'gpu':'0',
    'lr':0.0001,
    'num_neighbor':60,
    "entropy_point":0.1,
    "time_point":0.1,
    "count_point":0.1,  

    'max_expand_subgraph_size':6000,
    "output_edge_num":5000,
    'seed':1,
    'window_time':360000
}

