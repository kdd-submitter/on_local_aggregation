from argparse import ArgumentParser
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import sys
import dgl

import torch
import os

import ogb
from dgl.data import load_data
from dgl import DGLGraph
from ogb.nodeproppred import DglNodePropPredDataset


def create_synthetic_dataset(args,device = None):
    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)
    

    if 'ogbn' in args.dataset:
        dataset = DglNodePropPredDataset(name = args.dataset)
        ref_split_idx = dataset.get_idx_split()    
        ref_num_nodes = dataset[0][0].number_of_nodes()
        ref_labels = dataset[0][1].view(-1)
        ref_features = dataset[0][0].ndata['feat']
    else:
        dataset  = load_data(args)

        ref_split_idx = {'train' : np.where(dataset.train_mask)[0],
                         'valid' : np.where(dataset.val_mask)[0],
                         'test' : np.where(dataset.test_mask)[0]
                         }
        ref_num_nodes = DGLGraph(dataset.graph).number_of_nodes()
        ref_labels = torch.LongTensor(dataset.labels)        
        ref_features = torch.FloatTensor(dataset.features)

        

    N0 = args.syn_N0
    C= args.syn_C
    m = args.syn_m
    N = args.syn_N

    assert 0<=args.syn_homophily<=1, 'homophily must be in [0,1]'
    homophily = min(args.syn_homophily,1.0 - 1.0e-12) #Guard against division by zero later
    ##Homophily matrix generation. Assume C classes are arranged on a circle one unit length apart on the circle. Take exp(-min distance between 2 classes on the circle) as the weight
    ##for connecting nodes of these two classes together. Details in http://proceedings.mlr.press/v97/abu-el-haija19a/abu-el-haija19a-supp.pdf

    
    dist = np.abs(np.arange(C)[np.newaxis,:]   - np.arange(C)[:,np.newaxis])
    H = np.exp(-np.minimum(dist,C - dist))
    H[np.arange(C),np.arange(C)] = (H.sum(1)-1) * homophily / (1-homophily)
    
    assert (N % C) == 0
    synthetic_labels = np.arange(C).repeat(N // C)
    np.random.shuffle(synthetic_labels)
    synthetic_labels = torch.LongTensor(synthetic_labels)
    G = dgl.graph((np.arange(N0-1),np.arange(N0-1)+1),num_nodes = N)
    
    for idx in tqdm(range(N0,N)):
        degrees = G.in_degrees(np.arange(idx)) + G.out_degrees(np.arange(idx))
        compatibility= H[synthetic_labels[idx]][synthetic_labels[np.arange(idx)]]
        prob_unnormalized = (degrees * compatibility)
        prob  = prob_unnormalized / prob_unnormalized.sum()
        node_choices = np.random.choice(idx,size = m,replace = False,p = prob)
        G.add_edges(node_choices,idx)

    #Make bidirectional
    G.add_edges(G.all_edges()[1],G.all_edges()[0])

    
    #Add self-loop
    if args.self_loop:
        G = dgl.transform.add_self_loop(G)




    synthetic_train_indices = []
    synthetic_valid_indices = []
    synthetic_test_indices = []
    percent_25 = int(N//C * 0.25)
    percent_50 = int(N//C * 0.5)            
    for s_l in range(C):
        label_indices = torch.where(synthetic_labels == s_l)[0]
        assert len(label_indices) == N//C
        rand_perm = torch.randperm(N//C)
        synthetic_train_indices.append(label_indices[rand_perm[:percent_25]])
        synthetic_valid_indices.append(label_indices[rand_perm[percent_25:percent_50]])
        synthetic_test_indices.append(label_indices[rand_perm[percent_50:]])

    synthetic_train_indices,synthetic_valid_indices,synthetic_test_indices = map(lambda x : torch.cat(x,dim = 0),[synthetic_train_indices,synthetic_valid_indices,synthetic_test_indices])


    if args.syn_respect_original_split:
        def make_label_split_dict(labels,split_indices):
            D = {'train' : {},'valid' : {}, 'test' : {}}
            unique_labels = torch.unique(labels)
            for split_name in ['train','valid','test']:
                for l in unique_labels:
                    D[split_name][l.item()] = split_indices[split_name][(labels == l)[split_indices[split_name]]]
            return D

        ref_label_split_dict =  make_label_split_dict(ref_labels,ref_split_idx)       

        synthetic_split_idx = {'train' : synthetic_train_indices,
                               'valid' : synthetic_valid_indices,
                               'test' : synthetic_test_indices
                               }
        synthetic_label_split_dict = make_label_split_dict(synthetic_labels,synthetic_split_idx)
        
        

    synthetic_features = torch.zeros(N,ref_features.size(1))

    unique_ref_labels,unique_ref_labels_count = torch.unique(ref_labels,return_counts = True)
    compatible_ref_labels = unique_ref_labels[unique_ref_labels_count >= (N//C)]
    assert len(compatible_ref_labels) >= C , 'not enough labeled nodes in reference graph'

    compatible_ref_labels = compatible_ref_labels[torch.randperm(compatible_ref_labels.size(0))]
    ref_labels_begin_idx = 0
    for s_l in range(C):
        if ref_labels_begin_idx == len(compatible_ref_labels):
            raise Exception('mapping of indices failed')
        for n_trials,r_l in enumerate(compatible_ref_labels[ref_labels_begin_idx:]):
            r_l = r_l.item()
            print('trying to map ref label {} to synthetic label {}'.format(r_l,s_l))
            ref_label_mask = (ref_labels == r_l)
            if args.syn_respect_original_split:
                mapping_failed = False
                for split_name in ['train','valid','test']:
                    ref_indices = ref_label_split_dict[split_name][r_l]
                    synthetic_indices = synthetic_label_split_dict[split_name][s_l]
                    if len(ref_indices) < len(synthetic_indices):
                        print('not enough indices to map from {}/{} to {}/{} in split {}'.format(r_l,len(ref_indices),s_l,len(synthetic_indices),split_name))
                        mapping_failed = True
                        break
                    synthetic_features[synthetic_indices] = ref_features[ref_indices[torch.randperm(len(ref_indices))[:len(synthetic_indices)]]]
                if mapping_failed:
                    continue


            else:
                synthetic_label_locs = (synthetic_labels == s_l).nonzero()[0]

                ref_label_locs_all = torch.where(ref_labels == r_l)[0]
                ref_label_locs = ref_label_locs_all[torch.randperm(ref_label_locs_all.size(0))][:len(synthetic_label_locs)]


                synthetic_features[synthetic_label_locs] = ref_features[ref_label_locs]
            break
                
        ref_labels_begin_idx += n_trials+1
        print('mapping success')
    ##Get empirical homophily
    label_conns = synthetic_labels[G.all_edges()[0]] * C  + synthetic_labels[G.all_edges()[1]]
    pos,counts = np.unique(label_conns,return_counts = True)

    P = np.zeros(C*C)
    P[pos] = counts
    P = P.reshape((C,C))
    empirical_homophily = np.diag(P).sum() * 1.0 / P.sum()

    print("**empirical homophily is ",empirical_homophily)

    def to_mask(indx):      
        mask = torch.zeros(N,dtype = torch.bool)
        mask[indx] = 1
        return mask
    train_mask,val_mask,test_mask = map(to_mask,(synthetic_train_indices,synthetic_valid_indices,synthetic_test_indices))
    
    if args.syn_dump_path:
        torch.save({'graph' : G,
                    'labels' : synthetic_labels,
                    'features' : synthetic_features,
                    'train_mask' : train_mask,
                    'test_mask' : test_mask,
                    'val_mask' : val_mask,
                    'args' : args
                    }
                   ,args.syn_dump_path)

    device = torch.device('cpu') if device is None else device
    return (G.to(device),
            C,
            synthetic_features.to(device),
            synthetic_labels.to(device),
            train_mask.to(device),
            val_mask.to(device),
            test_mask.to(device))
        

if __name__ == '__main__':
    parser = ArgumentParser(description="Create synthetic dataset with configurable homophily")

    parser.add_argument(
        "--syn-N0",
        default=70,
        type=int,
        help="Size of core graph"
    )

    parser.add_argument(
        "--syn-C",
        default=10,
        type=int,
        help="Number of classes"
    )

    parser.add_argument(
        "--syn-m",
        default=6,
        type=int,
        help="Number of edges for each newly added node"
    )

    parser.add_argument(
        "--syn-N",
        default=10000,
        type=int,
        help="Number of nodes"
    )

    parser.add_argument(
        "--syn-homophily",
        type=float,
        default=0.5,
        help="Homophily coefficient"
    )
    

    parser.add_argument(
        "--syn-dump-path",
        type=str,
        default="",
        help="Path to file to dump the synthetic data to"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
        help="Base  dataset to use for generating the synthetic graphs",
    )

    parser.add_argument(
        "--self-loop",
        action='store_true',
        help="Add self loop"
    )

    parser.add_argument(
        "--syn-respect-original-split",
        action='store_true',
        help="Respect original train/valid/test split when copying features to synthetic"
    )
    
        
    parser.add_argument(
        "--split-seed",
        default=1,
        type=int,
        help="Seed to use for the generating the synthetic graph"
    )
    

    args= parser.parse_args()
    print(args)


    create_synthetic_dataset(args)
