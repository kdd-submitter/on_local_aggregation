from argparse import ArgumentParser
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import sys
import dgl

from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import os

from models import GNNMLP
from data_utils import load_dataset,ReadMixhopDataset
from create_synthetic_dataset import create_synthetic_dataset
from calculate_NIC import calculate_NIC

def save_results(state,  filename):
    """Saves checkpoint to disk"""
    directory = "runs/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


def infer_pass(model,all_data):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    graph,num_labels,features,labels,train_mask,val_mask,test_mask = all_data    
    with torch.no_grad():
        logits = model(graph, features)
        results_dict = {}
        for mask_name,mask in zip(['val','test'],[val_mask,test_mask]):
            targets = labels[mask]
            loss = criterion(logits[mask, :],labels[mask])
            accuracy = (logits[mask].argmax(1) == targets).float().mean().item()
            results_dict[mask_name] = [loss.item(),accuracy]
        return results_dict

def train_pass(args,model,optimizer,all_data):
    graph,num_labels,features,labels,train_mask,val_mask,test_mask = all_data
    model.train()
    criterion = nn.CrossEntropyLoss()

    logits = model(graph, features)
    loss = criterion(logits[train_mask],labels[train_mask])# + entropy_cost
    accuracy = (logits[train_mask].argmax(1) == labels[train_mask]).float().mean().item()    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(),accuracy




def main():
    parser = ArgumentParser(description="On local aggregation in heterophilic datasets")
    parser.add_argument(
        "-f", "--hidden-feat-repr-dims",
        default=[256,256],
        type=int,
        nargs="+",
        help="dimension of the hidden feature representations (no input or output)"
    )


    
    parser.add_argument(
        "--learnable-mixing",
        action='store_true',
        help="Learn mixing coefficients between gcn and mlp branch if both are enabled simultaneously"
    )


    parser.add_argument(
        "--small-train-split",
        action='store_true',
        help="Use a 10/10/80 percent train/validation/test split instead of default 60/20/20 percent"
    )


    
    parser.add_argument(
        "--use-sage",
        action='store_true',
        help="Use a graphsage layer instead of graphconv"
    )

    parser.add_argument(
        "--use-prelu",
        action='store_true',
        help="Use prelu activation instead of relu"
    )
    
    parser.add_argument(
        "--use-gat",
        action='store_true',
        help="Use a GAT layer instead of graphconv"
    )

    parser.add_argument(
        "--enable-mlp-branch",
        action='store_true',
        help="Enable MLP branch"
    )

    parser.add_argument(
        "--enable-gcn-branch",
        action='store_true',
        help="Enable GCN branch"
    )

    parser.add_argument(
        "--top-is-proj",
        action='store_true',
        help="Top layer is a projection(linear) layer and not a GNN layer"
    )
    
    parser.add_argument(
        "--gat-num-heads",
        default=1,
        type=int,
        help="Number of heads in GAT layer"
    )
    
        
    parser.add_argument(
        "--make-bidirectional",
        action='store_true',
        help="Make the graph bi-directional"
    )
    
    
    parser.add_argument(
        "-i", "--iterations",
        default=2000,
        type=int,
        help="number of training iterations"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="feature dropout probability"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="learning rate"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="weight decay"
    )
    

    
    parser.add_argument('--job-idx', default=0, type=int,
                    help='job index provided by the job manager')


    parser.add_argument(
        "--datasets-path",
        type=Path,
        default="./datasets/",
        help="Path to directory containing the geom-gcn graph datasets"
    )

    parser.add_argument(
        "--mixhop-dataset-path",
        type=str,
        default="",
        help="Path to Mixhop synthetic dataset"
    )
    
    parser.add_argument(
        "--custom-split-file", default="", type=str,
        help="file containing custom train,test,val splits. Only used to get the geom-gcn custom splits"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="squirrel",
        help="Dataset to use for training or as a source for node features when generating a synthetic dataset  : cora,citeseer, pubmed, film, wisconsin, cornell, texas,squirrel,chameleon, ogbn-*. Default : squirrel",
    )

    
    parser.add_argument(
        "--self-loop",
        action='store_true',
        help="Add self loop to the graph"
    )

    
    parser.add_argument(
        "--split-seed",
        default=1,
        type=int,
        help="Seed to use for various data splitting operations"
    )

    parser.add_argument(
        "--original-split",
        action='store_true',
        help="Uses the Kipf and Welling original split for cora, citeseer, and pubmed datasets"
    )

    parser.add_argument(
        "--homogeneous-split",
        action='store_true',
        help="Make sure the classes are balanced in the split, i.e, each class is equally represented in the training, validation, and testing splits"
    )


    parser.add_argument(
        "--use-synthetic-dataset",
        action='store_true',
        help="Use a synthetic dataset generated according to https://arxiv.org/abs/2006.11468"
    )


    parser.add_argument(
        "--syn-N0",
        default=70,
        type=int,
        help="Size of core graph when generating synthetic dataset"
    )

    parser.add_argument(
        "--syn-C",
        default=10,
        type=int,
        help="Number of classes when generating synthetc datasets"
    )

    parser.add_argument(
        "--syn-m",
        default=6,
        type=int,
        help="Number of edges for each newly added node when creating synthetic dataset"
    )

    parser.add_argument(
        "--syn-N",
        default=10000,
        type=int,
        help="Number of nodes in synthetically-generated graph"
    )

    parser.add_argument(
        "--syn-homophily",
        type=float,
        default=0.5,
        help="Homophily coefficient of synthetically generated graph"
    )
    

    parser.add_argument(
        "--syn-dump-path",
        type=str,
        default="",
        help="Path to file to dump the synthetically generated data to"
    )

    parser.add_argument(
        "--syn-respect-original-split",
        action='store_true',
        help="Respect original train/valid/test split when copying features from a real-world dataset to synthetic dataset"
    )
    

    
    args= parser.parse_args()
    print(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.use_synthetic_dataset:
        print('creating synthetic dataset based on ',args.dataset)
        graph,num_labels,features,labels,train_mask,val_mask,test_mask = create_synthetic_dataset(args,device)
    elif args.mixhop_dataset_path:
        graph,num_labels,features,labels,train_mask,val_mask,test_mask = ReadMixhopDataset(args.mixhop_dataset_path,device)
    else:
        graph,num_labels,features,labels,train_mask,val_mask,test_mask = load_dataset(args,device)

    print('train,val,test fractions',train_mask.float().mean(),val_mask.float().mean(),test_mask.float().mean())
        
    in_feature_dim = features.size(1)
    
    if args.make_bidirectional:
        graph = graph.to(torch.device('cpu'))
        graph_bi = dgl.to_bidirected(graph)
        for k,v in graph.ndata.items():
            graph_bi.ndata[k] = v
        graph = graph_bi.to(device)

    all_data = graph,num_labels,features,labels,train_mask,val_mask,test_mask
    feat_repr_dims = [in_feature_dim] + args.hidden_feat_repr_dims + [num_labels]        
    model = GNNMLP(feat_repr_dims,
                   enable_mlp = args.enable_mlp_branch,
                   enable_gcn = args.enable_gcn_branch,
                   learnable_mixing = args.learnable_mixing,
                   use_sage = args.use_sage,
                   use_gat = args.use_gat,
                   gat_num_heads = args.gat_num_heads,
                   top_is_proj = args.top_is_proj,
                   use_prelu = args.use_prelu,
                   dropout = args.dropout
                   ).to(device)
    print(model)
    print('# parameters ',sum([x.numel() for x in model.parameters()]))
    NIC = calculate_NIC(graph,labels,num_labels)
    print('NIC : ',NIC)
    optimizer = torch.optim.Adam(model.parameters(),args.lr,weight_decay = args.weight_decay)


    best_iter_loss = -1
    best_iter_accuracy = -1    
    best_val_loss = np.inf
    best_val_accuracy = 0.0
    val_loss_l = []
    val_acc_l = []
    test_loss_l = []
    test_acc_l = []
    for iter in range(args.iterations):
        loss_train,acc_train = train_pass(args,model,optimizer,all_data)
        results_dict = infer_pass(model,all_data)
        if results_dict['val'][0] < best_val_loss:
            best_iter_loss = iter
            best_val_loss = results_dict['val'][0]
            best_test_loss_loss,best_test_acc_loss = results_dict['test']

        if results_dict['val'][1] >= best_val_accuracy:
            best_iter_accuracy = iter
            best_val_accuracy = results_dict['val'][1]
            best_test_loss_acc,best_test_acc_acc = results_dict['test']
            
        #lr_scheduler.step()
        print('iter ',iter,' : train loss,accuracy :',loss_train,acc_train)
        print('iter ',iter,' : val loss,accuracy :',results_dict['val'])
        val_loss_l.append(results_dict['val'][0])
        val_acc_l.append(results_dict['val'][1])

        test_loss_l.append(results_dict['test'][0])
        test_acc_l.append(results_dict['test'][1])
    if hasattr(model,'mixing_coeffs'):
        print('mixing coeffs',model.mixing_coeffs)
    results_filename = 'GCNMLP' + '_' + repr(args.job_idx)        
    save_results({
        'test_acc' : best_test_acc_acc,
        'NIC' : NIC,
        'state_dict' : model.state_dict(),
        'args' : args},
        filename = results_filename)
        
    print('Test loss,accuracy at last epoch :',results_dict['test'])
    print('Test loss,accuracy at best validation loss epoch {} : {}/{}'.format(best_iter_loss,best_test_loss_loss,best_test_acc_loss))
    print('Test loss,accuracy at best validation accuracy epoch {} : {}/{}'.format(best_iter_accuracy,best_test_loss_acc,best_test_acc_acc))    

    


if __name__ == '__main__':
    main()
