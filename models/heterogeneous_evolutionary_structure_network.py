
import logging

import numpy as np
import torch


import modules
import modules.aggreation as aggreation


import logging
logging.getLogger().setLevel(logging.INFO)



class HONESTER(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat,num_layers, n_head,
                 max_expand_subgraph_size,
                 attn_mode='prod', use_time='time',
                 drop_out=0.1,  device = 'cpu',
                 gate = 'none',relation_num = 3
                 ):
        super(HONESTER, self).__init__()
        
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder

        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.device = device
        self.feat_dim = self.n_feat_th.shape[1]
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim 
        self.model_dim = self.feat_dim
        self.gate = gate 
        self.use_time = use_time
        self.relation_embed = torch.nn.Embedding(num_embeddings = relation_num ,embedding_dim = self.feat_dim  )
        self.max_expand_subgraph_size = max_expand_subgraph_size

        self.output_dim = self.feat_dim 
        self.merge_layer = modules.MergeLayer(self.output_dim, self.output_dim, self.output_dim, 1) 
        self.merge_entropy_layer = modules.MergeEntropyLayer(self.output_dim) 
       

        self.logger.info('Aggregation uses attention model')
        self.attn_model_list = torch.nn.ModuleList([aggreation.AttnModel(self.feat_dim, 
                                                            self.feat_dim, 
                                                            self.feat_dim,
                                                            attn_mode=attn_mode, 
                                                            n_head=n_head, 
                                                            drop_out=drop_out) for _ in range(num_layers)])
        

        self.time_encoder = modules.TimeEncode(expand_dim=self.n_feat_th.shape[1])
        self.entropy_encoder = modules.TimeEncode(expand_dim=self.n_feat_th.shape[1])
        self.relation_embed = modules.MultiHotLayer(dim1 = relation_num, outdim = self.feat_dim)

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder   

    
    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors):
        src_embed,weight = self.forward(src_idx_l, cut_time_l, self.num_layers, num_neighbors)

        target_embed,_ = self.forward(target_idx_l, cut_time_l, self.num_layers, num_neighbors)

        background_embed,_ = self.forward(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        
        pos_entropy_l = []
        neg_entropy_l = []
        for i in range(len(src_idx_l)):
            pos_entropy_l.append(self.ngh_finder.get_von_Neumann_entropy(src_idx_l[i],target_idx_l[i],cut_time_l[i]))
            neg_entropy_l.append(self.ngh_finder.get_von_Neumann_entropy(src_idx_l[i],background_idx_l[i],cut_time_l[i]))

        pos_entropy_l_th = torch.from_numpy(np.array(pos_entropy_l)).float().to(self.device)
        
        pos_entropy_l_th = torch.unsqueeze(pos_entropy_l_th, dim=1)


        neg_entropy_l_th = torch.from_numpy(np.array(neg_entropy_l)).float().to(self.device)
        
        neg_entropy_l_th = torch.unsqueeze(neg_entropy_l_th, dim=1)



        pos_entropy_embed = self.entropy_encoder(pos_entropy_l_th).view([-1,self.feat_dim])
        neg_entropy_embed = self.entropy_encoder(neg_entropy_l_th).view([-1,self.feat_dim])
        

        pos_score = self.merge_entropy_layer (src_embed, target_embed,pos_entropy_embed).squeeze(dim=-1)
        neg_score = self.merge_entropy_layer (src_embed, background_embed,neg_entropy_embed).squeeze(dim=-1)
        

        return pos_score.sigmoid(), neg_score.sigmoid()

    def forward(self, src_idx_l, cut_time_l, curr_layers, num_neighbors):   
            
        final_emb,weight = self.hetero_tem_conv(src_idx_l = src_idx_l, cut_time_l = cut_time_l, curr_layers = curr_layers,
                                                num_neighbors = num_neighbors,final_src_idx_l = src_idx_l, final_cut_time_l = cut_time_l)
        return final_emb,weight

    def hetero_tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors,final_src_idx_l,final_cut_time_l):
        assert(curr_layers >= 0)
        
    
        batch_size = len(src_idx_l)
        
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(self.device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(self.device)
        
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)
        
        if curr_layers == 0:
            return src_node_feat,0
        else:
            src_node_conv_feat,_f = self.hetero_tem_conv(src_idx_l = src_idx_l, 
                                           cut_time_l = cut_time_l,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors,final_src_idx_l = final_src_idx_l,final_cut_time_l = final_cut_time_l)


            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch,src_ngh_count_batch,src_ngh_relation_batch =\
            self.ngh_finder.get_heterogeneous_neighbor( final_src_idx_l = final_src_idx_l,
                                                        final_cut_time_l = final_cut_time_l,
                                                        src_idx_l = src_idx_l,
                                                        cut_time_l = cut_time_l,
                                                        num_neighbors=num_neighbors,
                                                        max_expand_subgraph_size = self.max_expand_subgraph_size
                                                        )
            # print(src_ngh_relation_batch.shape)
            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(self.device)
            
            src_ngh_relation_batch = torch.from_numpy(src_ngh_relation_batch).float().to(self.device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(self.device)
            
            
            
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() 
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() 

            src_ngh_node_conv_feat,_s = self.hetero_tem_conv(src_idx_l = src_ngh_node_batch_flat, 
                                                   cut_time_l = src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors,
                                                   final_src_idx_l = final_src_idx_l,
                                                   final_cut_time_l = final_cut_time_l)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
            
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)

            src_ngn_relation_feat = self.relation_embed(src_ngh_relation_batch)
            src_ngn_edge_feat = src_ngn_relation_feat
            
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src_node_conv_feat, 
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed, 
                                   src_ngn_edge_feat, 
                                   mask)
            return local, weight
        
