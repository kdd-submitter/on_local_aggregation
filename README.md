# on_local_aggregation
Code for the 2021 KDD submission 'On local aggregation in heterophilic graphs'. The main python executable is `main.py`. Results are saved under a `./runs/` directory created at the invocation directory. An invocation of `main.py` will save various accuracy metrics as well as the model parameters in the file `./runs/{model name}_{job idx}`. Accuracy figures as well as several diagnostics are also printed out. 

For the Cora, Citeseer, and Pubmed benchmarks, the graphs are downloaded automatically by the dgl library. The Actor, Wisconsin, Cornell, and Texas datasets are included in the repository and are located under the /datasets/ folder.

The Actor, Wisconsin, Cornell, Texas, Chameleon, and Squirrel  datasets are included in the repository and are located under the /datasets/ folder. The Mixhop synthetic graphs are included under the ./mixhop_data folder. The geom-gcn splits are included under the ./geom_gcn_splits/ folder.

### Required Packages
See requirements.txt for a complete list of required Python packages. Install required packages using the command:
```shell
pip install -r requirements.txt
```

### General usage

```shell
usage: main.py [-h] [-f HIDDEN_FEAT_REPR_DIMS [HIDDEN_FEAT_REPR_DIMS ...]]
               [--learnable-mixing] [--small-train-split] [--use-sage]
               [--use-prelu] [--use-gat] [--enable-mlp-branch]
               [--enable-gcn-branch] [--top-is-proj]
               [--gat-num-heads GAT_NUM_HEADS] [--make-bidirectional]
               [-i ITERATIONS] [--dropout DROPOUT] [--lr LR]
               [--weight-decay WEIGHT_DECAY] [--job-idx JOB_IDX]
               [--datasets-path DATASETS_PATH]
               [--mixhop-dataset-path MIXHOP_DATASET_PATH]
               [--custom-split-file CUSTOM_SPLIT_FILE] [--dataset DATASET]
               [--self-loop] [--split-seed SPLIT_SEED] [--original-split]
               [--homogeneous-split] [--use-synthetic-dataset]
               [--syn-N0 SYN_N0] [--syn-C SYN_C] [--syn-m SYN_M]
               [--syn-N SYN_N] [--syn-homophily SYN_HOMOPHILY]
               [--syn-dump-path SYN_DUMP_PATH] [--syn-respect-original-split]

On local aggregation in heterophilic datasets

optional arguments:
  -h, --help            show this help message and exit
  -f HIDDEN_FEAT_REPR_DIMS [HIDDEN_FEAT_REPR_DIMS ...], --hidden-feat-repr-dims HIDDEN_FEAT_REPR_DIMS [HIDDEN_FEAT_REPR_DIMS ...]
                        dimension of the hidden feature representations (no
                        input or output)
  --learnable-mixing    Learn mixing coefficients between gcn and mlp branch
                        if both are enabled simultaneously
  --small-train-split   Use a 10/10/80 percent train/validation/test split
                        instead of default 60/20/20 percent
  --use-sage            Use a graphsage layer instead of graphconv
  --use-prelu           Use prelu activation instead of relu
  --use-gat             Use a GAT layer instead of graphconv
  --enable-mlp-branch   Enable MLP branch
  --enable-gcn-branch   Enable GCN branch
  --top-is-proj         Top layer is a projection(linear) layer and not a GNN
                        layer
  --gat-num-heads GAT_NUM_HEADS
                        Number of heads in GAT layer
  --make-bidirectional  Make the graph bi-directional
  -i ITERATIONS, --iterations ITERATIONS
                        number of training iterations
  --dropout DROPOUT     feature dropout probability
  --lr LR               learning rate
  --weight-decay WEIGHT_DECAY
                        weight decay
  --job-idx JOB_IDX     job index provided by the job manager
  --datasets-path DATASETS_PATH
                        Path to directory containing the geom-gcn graph
                        datasets
  --mixhop-dataset-path MIXHOP_DATASET_PATH
                        Path to Mixhop synthetic dataset
  --custom-split-file CUSTOM_SPLIT_FILE
                        file containing custom train,test,val splits. Only
                        used to get the geom-gcn custom splits
  --dataset DATASET     Dataset to use for training or as a source for node
                        features when generating a synthetic dataset :
                        cora,citeseer, pubmed, film, wisconsin, cornell,
                        texas,squirrel,chameleon, ogbn-*. Default : squirrel
  --self-loop           Add self loop to the graph
  --split-seed SPLIT_SEED
                        Seed to use for various data splitting operations
  --original-split      Uses the Kipf and Welling original split for cora,
                        citeseer, and pubmed datasets
  --homogeneous-split   Make sure the classes are balanced in the split, i.e,
                        each class is equally represented in the training,
                        validation, and testing splits
  --use-synthetic-dataset
                        Use a synthetic dataset generated according to
                        https://arxiv.org/abs/2006.11468
  --syn-N0 SYN_N0       Size of core graph when generating synthetic dataset
  --syn-C SYN_C         Number of classes when generating synthetc datasets
  --syn-m SYN_M         Number of edges for each newly added node when
                        creating synthetic dataset
  --syn-N SYN_N         Number of nodes in synthetically-generated graph
  --syn-homophily SYN_HOMOPHILY
                        Homophily coefficient of synthetically generated graph
  --syn-dump-path SYN_DUMP_PATH
                        Path to file to dump the synthetically generated data
                        to
  --syn-respect-original-split
                        Respect original train/valid/test split when copying
                        features from a real-world dataset to synthetic
                        dataset

```

### Performing the experiments in the paper
The YAML files under the all_experiments folder contains list of the invocation commands needed to run the experiments described in the paper. See the README file in that folder.
