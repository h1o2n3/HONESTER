from re import L
import numpy as np



class HeteroNeighborFinder:

    def __init__(self, adj_list, uniform ,entropy_point, time_point,count_point,output_edge_num,window_time):
        self.node_num = len(adj_list)
        
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)  
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.off_set_l = off_set_l
        self.edge_num = len(self.edge_idx_l)
        self.uniform = uniform
        self.max_entropy = 9375
        self.c_param = 100000
        self.neighbor_dict = {}
        self.count_point = count_point
        self.entropy_point = entropy_point
        self.time_point = time_point
        self.output_edge_num = output_edge_num
        self.window_time = window_time
    def init_off_set(self, adj_list):

            n_idx_l = []
            n_ts_l = []
            e_idx_l = []
            off_set_l = [0]
            length = self.node_num 

            for i in range(length):
                curr = adj_list[i]
                curr = sorted(curr, key=lambda x: x[1],reverse = False)
                n_idx_l.extend([x[0] for x in curr])
                e_idx_l.extend([x[1] for x in curr])
                n_ts_l.extend([x[2] for x in curr])
            
                
                off_set_l.append(len(n_idx_l))
            n_idx_l = np.array(n_idx_l)
            n_ts_l = np.array(n_ts_l)
            e_idx_l = np.array(e_idx_l)
            off_set_l = np.array(off_set_l)

            assert(len(n_idx_l) == len(n_ts_l))
            assert(off_set_l[-1] == len(n_ts_l))
            
            return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before_degree(self, src_idx, cut_time):

        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        

        left = -1
        right = len(neighbors_idx)
        
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t <= cut_time:
                left = mid
            else:
                right = mid
        return neighbors_idx[:left+1], neighbors_e_idx[:left+1], neighbors_ts[:left+1],left+1,neighbors_idx[:left+1]

    def find_before(self, src_idx, cut_time):

        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            assert("neighbors find error")
            

        left = -1
        right = len(neighbors_idx) 
        
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t <= cut_time:
                left = mid
            else:
                right = mid
        return neighbors_idx[:left+1], neighbors_e_idx[:left+1], neighbors_ts[:left+1]

    def get_von_Neumann_entropy_each(self,du,dv,u_node_l,v_node_l,v_idx,u_idx):
        if du == 0 :
            du = 1
        if dv == 0 :
            dv = 1
        subgraph_node_num = len(
            set(
                list(np.concatenate([u_node_l,v_node_l,np.array([v_idx]),np.array([u_idx])],axis = 0))
                )
            )
        entropy = ((1/subgraph_node_num)**2) * (du+dv+1) / (du*(du+1) * dv * (dv+1)) * self.c_param
        entropy = round(entropy,4)

        return entropy      
    
    def get_heterogeneous_neighbor_find_before(self, src_idx, cut_time):
        ngh_idx, ngh_eidx, ngh_ts,du,u_node_l  = self.find_before_degree(src_idx, cut_time)

        length_ngh = len(ngh_idx)
        

        length = length_ngh
        ngh_idx = ngh_idx.tolist()
        neighor_list = []

        
        if length == 0 : 
            return [],0
        for i in range(length):
            
            v_idx = ngh_idx[i]
            
            if v_idx not in self.already_ex_node_list:
                count = ngh_idx.count(v_idx)
                time = ngh_ts[i]
                _,_,_,dv,v_node_l = self.find_before_degree(v_idx,time)

                entropy = self.get_von_Neumann_entropy_each(du = du,dv = dv,u_node_l = u_node_l,v_node_l = v_node_l
                ,v_idx = v_idx,u_idx = src_idx)            
                if entropy < self.max_entropy:                   
                    if src_idx <= ngh_idx[i]:
                        neighor_list.append((src_idx,ngh_idx[i],ngh_eidx[i],time,count,entropy)) 
                    else:
                        neighor_list.append((ngh_idx[i],src_idx,ngh_eidx[i],time,count,entropy))

        return neighor_list,ngh_idx 

    
    def set_candidate_edge_set_list(self,src_idx_l,cut_time_l,max_expand_subgraph_size):
        hop = 2
        f_neighor_list_all = []
        for node_index in range(len(src_idx_l)):                
            
            expand_node_idx = src_idx_l[node_index]
            query_time =  cut_time_l[node_index]
            
            f_neighor_list = []
            
            self.already_ex_node_list = []
            expand_node_list = [(expand_node_idx,query_time)]
            
            
            
            cut_time = query_time
            for i in range(hop):  
                
                neibor_hop_list = []
                while len(expand_node_list) != 0 and len(f_neighor_list) < max_expand_subgraph_size: 
                    src_node = expand_node_list.pop(0)
                    src_idx = src_node[0]
                    
                    if src_idx not in self.already_ex_node_list:
                        self.already_ex_node_list.append(src_idx)
                        
                        
                        
                        neighor_list,ngh_idx = self.get_heterogeneous_neighbor_find_before(src_idx, cut_time)
                        
                        f_neighor_list = f_neighor_list + neighor_list
                        if neighor_list != []:
                            max_cut_time = max(neighor_list,key=lambda x:x[-3])[-3]
                            neibor_hop_list = neibor_hop_list + [(x,max_cut_time) for x in ngh_idx]

                neibor_hop_list = list(set(neibor_hop_list))
                expand_node_list = neibor_hop_list

            if len(f_neighor_list) == 0:
                f_neighor_list_all.append([ (src_idx_l[node_index], src_idx_l[node_index],0,0.0,0,0)])
            else:
                f_neighor_list_all.append(f_neighor_list)
           
        return f_neighor_list_all



    def h_edge_generation(self,edge_list):

        length = len(edge_list)
        for j in range(len(edge_list)):
            edge_list[j] = (edge_list[j] + (np.array([0,0,0]),))
        
        edge_list.sort(key = lambda x:x[-3],reverse=True)

        for i in range(int(length*self.count_point+1)):
            # print(i)
            edge_list[i][-1][0] = 1

        edge_list.sort(key = lambda x:x[-2],reverse=True)
        for i in range(int(length*self.entropy_point+1)):
            
            edge_list[i][-1][1] = 1    

        edge_list.sort(key = lambda x:x[-4],reverse=True)
        for i in range(int(length*self.time_point+1)):

            edge_list[i][-1][2] = 1            


        return edge_list

    def list2dict(self,src_idx_l,cut_time_l,f_neighor_list_all):
        for i in range(len(f_neighor_list_all)):
            f_neighor_list = f_neighor_list_all[i]
            src_idx = src_idx_l[i]
            cut_time = cut_time_l[i]
            if src_idx not in self.neighbor_dict:
                self.neighbor_dict[src_idx] = {}
            if cut_time not in self.neighbor_dict[src_idx]:
                self.neighbor_dict[src_idx][cut_time] = {}    
                
            

            new_edge_list = self.h_edge_generation(f_neighor_list)  


            for new_edge in new_edge_list:
                head_node = new_edge[0]
                tail_node = new_edge[1]


                if head_node not in self.neighbor_dict[src_idx][cut_time]:
                    self.neighbor_dict[src_idx][cut_time][head_node] = []
                
                self.neighbor_dict[src_idx][cut_time][head_node].append(new_edge[1:])

                if tail_node not in self.neighbor_dict[src_idx][cut_time]:
                    self.neighbor_dict[src_idx][cut_time][tail_node] = []
                
                self.neighbor_dict[src_idx][cut_time][tail_node].append((new_edge[0:1]+new_edge[2:]))
                                    
    def vNE_neighborhood_sampling(self,f_neighor_list_all):
        # print(f_neighor_list_all)  
        output_edge_list_all = [] 
        for k in range(len(f_neighor_list_all)):
            f_neighor_list = f_neighor_list_all[k]
            # print(f_neighor_list)
            time_list = [x[-3] for x in f_neighor_list]
            time_max = max(time_list)
            time_min = min(time_list)

            f_neighor_list.sort(key = lambda x:x[-3])
            candidate_length = len(f_neighor_list)
            time_length = time_max - time_min
            window_num = time_length // self.window_time + 1

            window_list = [[] for i in range(int(window_num))]
            output_edge_list = []
            edge_index = 0
            for j in range(len(window_list)):
                max_window_time = time_min + self.window_time*(j+1)
                min_window_time = time_min + self.window_time*(j)
                if edge_index == candidate_length:
                    break
                while edge_index < candidate_length:
                    if f_neighor_list[edge_index][-3]<=max_window_time and  f_neighor_list[edge_index][-3]>=min_window_time:
                        window_list[j].append(f_neighor_list[edge_index])
                        edge_index += 1
                    else:
                        break
                topk = int(self.output_edge_num * len(window_list[j])/ candidate_length) +1
                window_list[j].sort(key= lambda x:x[-1],reverse=True)
                output_edge_list = output_edge_list + window_list[j][:topk]


            output_edge_list_all.append(output_edge_list)

        return output_edge_list_all
        

    
    def set_heterogeneous_neighbor(self,src_idx_l,cut_time_l,max_expand_subgraph_size):


        candidate_edge_set_list = self.set_candidate_edge_set_list(src_idx_l,cut_time_l,max_expand_subgraph_size)
        output_edge_list_all =  self.vNE_neighborhood_sampling(candidate_edge_set_list)

        self.list2dict(src_idx_l,cut_time_l,output_edge_list_all)
    
    def find_tree(self,i,num_neighbors):
        idx = 0
        while i > (idx+1) * num_neighbors:
            idx += 1
        return idx

    def get_heterogeneous_neighbor(self,final_src_idx_l,final_cut_time_l,src_idx_l,cut_time_l,num_neighbors,max_expand_subgraph_size):

        length = len(src_idx_l)
        out_ngh_node_batch = np.zeros((length, num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((length, num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((length, num_neighbors)).astype(np.int32)
        out_ngh_count_batch = np.zeros((length, num_neighbors)).astype(np.int32)
        out_ngh_relation_batch = np.zeros((length, num_neighbors,3)).astype(np.int32)

        for i,node in enumerate(src_idx_l) :
            if node == 0:
                continue
            
            if len(final_src_idx_l) == len(src_idx_l):
                idx = i
            else:
                idx = self.find_tree(i,num_neighbors)
            final_src_idx = final_src_idx_l[idx] 
            final_cut_time = final_cut_time_l[idx]
            if final_src_idx in self.neighbor_dict and final_cut_time in self.neighbor_dict[final_src_idx]:
                
                if node in self.neighbor_dict[final_src_idx][final_cut_time]:
                    neibor = self.neighbor_dict[final_src_idx][final_cut_time][node] 
                    node_l = [ ]
                    eidx_l = []
                    t_l = []
                    count_l = []
                    relation_l = []
                    entropy_l = []
                    for x in neibor:
                        node_l.append(x[0])  
                        eidx_l.append(x[1])  
                        t_l.append(x[2])  
                        count_l.append(x[3])  
                        entropy_l.append(x[4])  
                        relation_l.append(x[5])
                    length_neibor = len(node_l[:num_neighbors])
                    if length_neibor > 0:
                        out_ngh_node_batch[i, num_neighbors - length_neibor:] = node_l[:num_neighbors]
                        out_ngh_eidx_batch[i,  num_neighbors - length_neibor:] = eidx_l[:num_neighbors]         
                        out_ngh_t_batch[i, num_neighbors - length_neibor:] = t_l[:num_neighbors]
                        out_ngh_count_batch[i,  num_neighbors - length_neibor:] = count_l[:num_neighbors]  
                        out_ngh_relation_batch[i,  num_neighbors - length_neibor:,:] = relation_l[:num_neighbors]  
            else:
                
                self.set_heterogeneous_neighbor([final_src_idx],[final_cut_time],max_expand_subgraph_size)
                if node in self.neighbor_dict[final_src_idx][final_cut_time]:
                    neibor = self.neighbor_dict[final_src_idx][final_cut_time][node] 
                    node_l = [ ]
                    eidx_l = []
                    t_l = []
                    count_l = []
                    relation_l = []
                    entropy_l = []
                    for x in neibor:
                        node_l.append(x[0])  
                        eidx_l.append(x[1])  
                        t_l.append(x[2])  
                        count_l.append(x[3])  
                        entropy_l.append(x[4])  
                        relation_l.append(x[5])
                    length_neibor = len(node_l[:num_neighbors])

                    if length_neibor > 0:
                        out_ngh_node_batch[i, num_neighbors - length_neibor:] = node_l[:num_neighbors]
                        out_ngh_eidx_batch[i,  num_neighbors - length_neibor:] = eidx_l[:num_neighbors]         
                        out_ngh_t_batch[i, num_neighbors - length_neibor:] = t_l[:num_neighbors]
                        out_ngh_count_batch[i,  num_neighbors - length_neibor:] = count_l[:num_neighbors]  
                        out_ngh_relation_batch[i,  num_neighbors - length_neibor:,:] = relation_l[:num_neighbors]   

        return  out_ngh_node_batch,out_ngh_eidx_batch,out_ngh_t_batch,out_ngh_count_batch,out_ngh_relation_batch   


    def get_von_Neumann_entropy(self,u,v,t):
        _,_,_,dv,v_node_l = self.find_before_degree(v,t)
        _,_,_,du,u_node_l = self.find_before_degree(u,t)

        entropy = self.get_von_Neumann_entropy_each(du = du,dv = dv,u_node_l = u_node_l,v_node_l = v_node_l
        ,v_idx = v,u_idx = u)

        return entropy
