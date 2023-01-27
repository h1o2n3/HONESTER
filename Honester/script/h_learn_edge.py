import math
import logging
import time
import sys
import joblib
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import sys,os
import time
sys.path.append(os.getcwd()+'/')
sys.path.append(os.getcwd()+'/script/')
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import models
import modules
import utils  
import ledge_configuration
from sacred import Experiment


ex = Experiment(save_git_info=False)

@ex.config
def my_config():
    parameter_dict = ledge_configuration.PARAMETER_DICT





@ex.capture
def eval_one_epoch(hint, honester, sampler, src, dst, ts, label,parameter_dict ,flag):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        honester = honester.eval()
        TEST_BATCH_SIZE=30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            if s_idx == e_idx:
                continue   
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)
            pos_prob, neg_prob = honester.contrast(src_idx_l = src_l_cut,target_idx_l =  dst_l_cut, background_idx_l = dst_l_fake,
             cut_time_l = ts_l_cut, num_neighbors = parameter_dict['num_neighbor']
                    )

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

                                
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_auc)

def seed_torch(sys_seed):
    sys_seed = int(sys_seed)
    random.seed(sys_seed)
    os.environ['PYTHONHASHSEED'] = str(sys_seed)
    np.random.seed(sys_seed)
    torch.manual_seed(sys_seed)
    torch.cuda.manual_seed(sys_seed)
    torch.cuda.manual_seed_all(sys_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
            
@ex.automain
def main(parameter_dict,_run):
    MODEL_SAVE_PATH = './saved_models/{dataset}/'.format(dataset = parameter_dict['data_name'])
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    modeltime = time.strftime('%Y-%m-%d-%H_%M_%S',time.localtime(time.time()))
    MODEL_SAVE_NAME = MODEL_SAVE_PATH + '{dataset}_{time}.pth'.format(dataset = parameter_dict['data_name'],time = modeltime)
    get_epoch_model_save_name = lambda epoch: MODEL_SAVE_PATH + f'{epoch}'+'_{dataset}_{time}.pth'.format(dataset = parameter_dict['data_name'],time = modeltime)
    
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("INFO")
    ex.logger = logger

    seed_torch(parameter_dict['seed'])
    g_df = pd.read_csv(ledge_configuration.PATH+'/data/{}/ml_{}.csv'.format(parameter_dict['data_name'],parameter_dict['data_name']))
    e_feat = np.load(ledge_configuration.PATH+'/data/{}/ml_{}.npy'.format(parameter_dict['data_name'],parameter_dict['data_name']))
    n_feat = np.load(ledge_configuration.PATH+'/data/{}/ml_{}_node.npy'.format(parameter_dict['data_name'],parameter_dict['data_name']))

    data_feature_path = ledge_configuration.PATH+'/data_feature/{}/seed_{}/'.format(
        parameter_dict['data_name'],parameter_dict['seed'])


    
    train_adj_path = data_feature_path +'{}_train_adj_list_{}.pkl'.format(
        parameter_dict['data_name'],parameter_dict['seed'])

    full_adj_path = data_feature_path + '{}_full_adj_list_{}.pkl'.format(
        parameter_dict['data_name'],parameter_dict['seed'])

        

    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values

    max_idx = max(src_l.max(), dst_l.max())
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)

    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    train_node_set = set(train_src_l).union(train_dst_l)
    assert(len(train_node_set - mask_node_set) == len(train_node_set))
    new_node_set = total_node_set - train_node_set

    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
    nn_val_flag = valid_val_flag * is_new_node_edge
    nn_test_flag = valid_test_flag * is_new_node_edge

    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]

    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]
    nn_val_src_l = src_l[nn_val_flag]
    nn_val_dst_l = dst_l[nn_val_flag]
    nn_val_ts_l = ts_l[nn_val_flag]
    nn_val_label_l = label_l[nn_val_flag]

    nn_test_src_l = src_l[nn_test_flag]
    nn_test_dst_l = dst_l[nn_test_flag]
    nn_test_ts_l = ts_l[nn_test_flag]
    nn_test_label_l = label_l[nn_test_flag]

    if os.path.exists(train_adj_path):
        ex.logger.info("successful load in {}.......".format(train_adj_path))
        adj_list = joblib.load(train_adj_path)
    else:
        os.makedirs(data_feature_path)
        ex.logger.info("never load {}, start building......".format(train_adj_path))
        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
            adj_list[src].append((dst, eidx, ts))
            adj_list[dst].append((src, eidx, ts))
        joblib.dump(adj_list,train_adj_path)

    train_ngh_finder = modules.HeteroNeighborFinder(adj_list,
    uniform=False,
    entropy_point=parameter_dict['entropy_point'], time_point= parameter_dict['time_point'],count_point = parameter_dict['count_point'],
    output_edge_num = parameter_dict["output_edge_num"],window_time=  parameter_dict["window_time"])

    if os.path.exists(full_adj_path):
        ex.logger.info("successful load in {}......".format(full_adj_path))
        full_adj_list = joblib.load(full_adj_path)
    else:
        ex.logger.info("never load {}, start building......".format(full_adj_path))
        full_adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            full_adj_list[src].append((dst, eidx, ts))
            full_adj_list[dst].append((src, eidx, ts))
        joblib.dump(full_adj_list,full_adj_path)
    
    full_ngh_finder = modules.HeteroNeighborFinder(full_adj_list, uniform=False,
    entropy_point=parameter_dict['entropy_point'], time_point= parameter_dict['time_point'],count_point = parameter_dict['count_point'],
    output_edge_num = parameter_dict["output_edge_num"],window_time=  parameter_dict["window_time"])

    train_rand_sampler = utils.RandEdgeSampler(train_src_l, train_dst_l)
    val_rand_sampler = utils.RandEdgeSampler(src_l, dst_l)
    test_rand_sampler = utils.RandEdgeSampler(src_l, dst_l)
    nn_test_rand_sampler = utils.RandEdgeSampler(nn_test_src_l, nn_test_dst_l)

    if parameter_dict['gpu'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(parameter_dict['gpu']))
    honester = models.HONESTER(ngh_finder = train_ngh_finder, n_feat = n_feat, e_feat = e_feat,
                num_layers=parameter_dict['n_layer'],attn_mode=parameter_dict['attn_mode'],
                n_head=parameter_dict['n_head'], drop_out=parameter_dict['drop_out'], 
                device= device,
                max_expand_subgraph_size = parameter_dict['max_expand_subgraph_size'])
    optimizer = torch.optim.Adam(honester.parameters(), lr=parameter_dict['lr'])
        
    criterion = torch.nn.BCELoss()
    honester = honester.to(device)

    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / parameter_dict['batch'])

    ex.logger.info('num of training instances: {}'.format(num_instance))
    ex.logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list) 

    early_stopper = utils.EarlyStopMonitor()
    for epoch in range(parameter_dict['n_epoch']):
        honester.ngh_finder = train_ngh_finder
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        ex.logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch)):
                                    
            s_idx = k * parameter_dict['batch']
            e_idx = min(num_instance - 1, s_idx + parameter_dict['batch'])
            if s_idx == e_idx:
                continue   
            
            src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
            

            ts_l_cut = train_ts_l[s_idx:e_idx]
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

            optimizer.zero_grad()
            honester = honester.train()  
            pos_prob, neg_prob = honester.contrast(src_idx_l = src_l_cut, target_idx_l =  dst_l_cut, background_idx_l = dst_l_fake,
             cut_time_l = ts_l_cut, num_neighbors = parameter_dict['num_neighbor']
                    )
            loss1 = criterion(pos_prob, pos_label)
            loss2 = criterion(neg_prob, neg_label)
            loss =  loss1 + loss2
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                honester = honester.eval()
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))
                        
                honester.ngh_finder = full_ngh_finder
                val_acc, val_ap, val_auc = eval_one_epoch('val for old nodes', honester, val_rand_sampler, val_src_l, 
        val_dst_l, val_ts_l, val_label_l,parameter_dict,'_val_old_')

        nn_val_acc, nn_val_ap, nn_val_auc = eval_one_epoch('val for new nodes', honester, val_rand_sampler, nn_val_src_l, 
        nn_val_dst_l, nn_val_ts_l, nn_val_label_l,parameter_dict ,'_val_new_')
            
        ex.logger.info('epoch: {}:'.format(epoch))
        ex.logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        ex.logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), np.mean(val_auc),np.mean( nn_val_auc)))
        ex.logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), np.mean(val_ap), np.mean(nn_val_ap)))
        
        if early_stopper.early_stop_check(val_ap):
            ex.logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            ex.logger.info('Loading the best model at epoch {}'.format(early_stopper.best_epoch))
            honester.eval()
            break
        else:
            torch.save(honester.state_dict(), get_epoch_model_save_name(epoch=epoch))


    honester.ngh_finder = full_ngh_finder
    test_acc, test_ap, test_auc = eval_one_epoch('test for old nodes', honester, test_rand_sampler, test_src_l, 
    test_dst_l, test_ts_l, test_label_l,parameter_dict ,'_test_old_')

    nn_test_acc, nn_test_ap , nn_test_auc = eval_one_epoch('test for new nodes', honester, nn_test_rand_sampler, nn_test_src_l, 
    nn_test_dst_l, nn_test_ts_l, nn_test_label_l,parameter_dict ,'_test_new_')
    _run.info = {
        'old_acc':[np.mean(test_acc),np.mean(test_auc),np.mean(test_ap)],
        'new_acc':[np.mean(nn_test_acc), np.mean(nn_test_auc), np.mean(nn_test_ap)]
                    } 

    ex.logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(np.mean(test_acc), np.mean(test_auc), np.mean(test_ap)))
    ex.logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(np.mean(nn_test_acc), np.mean(nn_test_auc), np.mean(nn_test_ap)))
    
    ex.logger.info('Saving HONESTER model')
    
    torch.save(honester.state_dict(), MODEL_SAVE_NAME)
    ex.logger.info('HONESTER models saved')

if __name__ == '__main__':
        main()