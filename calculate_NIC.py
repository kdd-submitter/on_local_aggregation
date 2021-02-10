import numpy as np
import dgl
import torch

def pairwise_dist_bound(node_degrees_probs,class_probabilities,compatibility,upper_bound = False):
    bound = 0
    max_deg = len(node_degrees_probs[0])
    for i in range(len(class_probabilities)):
        acc = 0
        for j in range(len(class_probabilities)):
            dist = 0 
            for deg in range(max_deg):
                if upper_bound:
                    dist +=  deg * deg_prob * (np.log(compatibility[i] / compatibility[j]) * compatibility[i]).sum() 
                else:
                    dist += np.sqrt(node_degrees_probs[i][deg] * node_degrees_probs[j][deg]) * (np.sqrt(compatibility[i] * compatibility[j]).sum())**deg
            if not(upper_bound):
                dist  = -np.log(dist)
            acc += class_probabilities[j] * np.exp(-dist)
        bound += class_probabilities[i] * np.log(acc)
    return -bound


def calculate_NIC(graph,labels,num_labels):
    graph = graph.to(torch.device('cpu'))
    all_edges = torch.stack(graph.all_edges())

    all_edges = all_edges.numpy()
    labels = labels.cpu()
    

    C = num_labels
    G = graph
    label_conns = labels[G.all_edges()[1]] * C  + labels[G.all_edges()[0]]
    pos,counts = np.unique(label_conns,return_counts = True)

    P = np.zeros(C*C)
    P[pos] = counts
    P = P.reshape((C,C))
    compatibility = P / P.sum(1,keepdims = True)
    compatibility[compatibility == 0] = 1.0e-10


    degs = graph.in_degrees()
    node_degs_probs = [np.zeros(degs.max()+1) for _ in range(num_labels)]
    for i in range(num_labels):
        label_degs,label_degs_count = np.unique(degs[labels == i],return_counts = True)
        label_degs_probs = label_degs_count / label_degs_count.sum()
        node_degs_probs[i][label_degs] = label_degs_probs

    lbl_idx,lbl_count = np.unique(labels,return_counts = True)
    class_probabilities = np.zeros(num_labels)
    class_probabilities[lbl_idx] = lbl_count / lbl_count.sum()



    NIC = pairwise_dist_bound(node_degs_probs,class_probabilities,compatibility)
    return NIC
