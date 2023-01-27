import sys,os
sys.path.append(os.getcwd()+'/')
sys.path.append(os.getcwd()+'/script/')
from script import h_learn_edge
import argparse
import numpy as np
import json
from sacred.observers import FileStorageObserver
import ledge_configuration
np.seterr(all='raise') 
parser = argparse.ArgumentParser('Interface for HONESTER experiments on link predictions')
parser.add_argument('-g', type=str, default='0', help='idx for the gpu to use')
parser.add_argument('-b', type=int, default='512', help='idx for the gpu to use')
parser.add_argument('-s', type=str, default='w', help='idx for the gpu to use')


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

parameter_dict = { }

with open('./update_json/'+ args.s + '.json','r',encoding='utf8') as fp:
    parameter_dict = json.load(fp)


parameter_dict['gpu'] = args.g
parameter_dict['batch'] = args.b

update_dict = {'parameter_dict':parameter_dict
                }
update_dict['seed'] = 785538553

if 'data_name' in parameter_dict:
    h_learn_edge.ex.observers.append(FileStorageObserver('./results/HONESTER/'+parameter_dict['data_name']))
else:
    h_learn_edge.ex.observers.append(FileStorageObserver('./results/HONESTER/'+ledge_configuration.PARAMETER_DICT['data_name']))
r = h_learn_edge.ex.run(config_updates=update_dict)

