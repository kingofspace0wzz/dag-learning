import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.linalg as slin
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
import glob
import re
import pickle
import math
from torch.optim.adam import Adam
import cdt
import argparse

#==============
# Gran-DAG
#==============
class DataManagerFile(object):
    def __init__(self, file_path, i_dataset, train_samples=0.8, test_samples=None, train=True, normalize=False,
                 mean=None, std=None, random_seed=42):
        """
        Parameters:
        -----------
        train_samples: uint or float, default=0.8
            If float, specifies the proportion of data used for training and the rest is used for testing. If an
            integer, specifies the exact number of examples to use for training.
        test_samples: uint, default=None
            Specifies the number of examples to use for testing. The default value uses all examples that are not used
            for training.

        """
        self.random = np.random.RandomState(random_seed)

        # Load the graph
        adjacency = np.load(os.path.join(file_path, "DAG{}.npy".format(i_dataset)))
        self.adjacency = torch.as_tensor(adjacency).type(torch.Tensor)

        # Load data
        self.data_path = os.path.join(file_path, "data{}.npy".format(i_dataset))
        data = np.load(self.data_path)

        # Determine train/test partitioning
        if isinstance(train_samples, float):
            train_samples = int(data.shape[0] * train_samples)
        if test_samples is None:
            test_samples = data.shape[0] - train_samples
        assert train_samples + test_samples <= data.shape[0], "The number of examples to load must be smaller than " + \
            "the total size of the dataset"

        # Shuffle and filter examples
        shuffle_idx = np.arange(data.shape[0])
        self.random.shuffle(shuffle_idx)
        data = data[shuffle_idx[: train_samples + test_samples]]

        # Train/test split
        if not train:
            if train_samples == data.shape[0]: # i.e. no test set
                self.dataset = None
            else:
                self.dataset = torch.as_tensor(data[train_samples: train_samples + test_samples]).type(torch.Tensor)
        else:
            self.dataset = torch.as_tensor(data[: train_samples]).type(torch.Tensor)

        # Normalize data
        self.mean, self.std = mean, std
        if normalize:
            if self.mean is None or self.std is None:
                self.mean = torch.mean(self.dataset, 0, keepdim=True)
                self.std = torch.std(self.dataset, 0, keepdim=True)
            self.dataset = (self.dataset - self.mean) / self.std

        self.num_samples = self.dataset.size(0)

    def sample(self, batch_size):
        sample_idxs = self.random.choice(np.arange(int(self.num_samples)), size=(int(batch_size),), replace=False)
        samples = self.dataset[torch.as_tensor(sample_idxs).long()]
        return samples, torch.ones_like(samples)  # second output is mask (for intervention in the future)

#================
# DNN-DAG
#================
def read_BNrep(args):
    '''load results from BN repository'''

    if args.data_filename == 'alarm':
        data_dir = os.path.join(args.data_dir, 'alarm/')
    elif args.data_filename == 'child':
        data_dir = os.path.join(args.data_dir, 'child/')
    elif args.data_filename =='hail':
        data_dir = os.path.join(args.data_dir, 'hail/')
    elif args.data_filename =='alarm10':
        data_dir = os.path.join(args.data_dir, 'alarm10/')
    elif args.data_filename == 'child10':
        data_dir = os.path.join(args.data_dir, 'child10/')
    elif args.data_filename == 'pigs':
        data_dir = os.path.join(args.data_dir, 'pigs/')

    all_data = dict()
    # read text files
    file_pattern = data_dir +"*_s*_v*.txt"
    all_files = glob.iglob(file_pattern)
    for file in all_files:
        match = re.search('/([\w]+)_s([\w]+)_v([\w]+).txt', file)
        dataset, samplesN, version = match.group(1), match.group(2),match.group(3)

        # read file
        data = np.loadtxt(file, skiprows =0, dtype=np.int32)
        if samplesN not in all_data:
            all_data[samplesN] = dict()

        all_data[samplesN][version] = data

    # read ground truth graph
    from os import listdir

    file_pattern = data_dir + "*_graph.txt"
    files = glob.iglob(file_pattern)
    for f in files:
        graph = np.loadtxt(f, skiprows =0, dtype=np.int32)

    return all_data, graph # in dictionary

def df_to_tensor(df):
    return torch.from_numpy(df.values).float()

def load_data_discrete(args, batch_size=1000, suffix='', debug = False):
    #  # configurations
    n, d = args.data_sample_size, args.data_variable_size
    graph_type, degree, sem_type = args.graph_type, args.graph_degree, args.graph_sem_type

    if args.data_type == 'synthetic':
        # generate data
        G = simulate_random_dag(d, degree, graph_type)
        X = simulate_sem(G, n, sem_type)

    elif args.data_type == 'discrete':
        # get benchmark discrete data
        if args.data_filename.endswith('.pkl'):
            with open(os.path.join(args.data_dir, args.data_filename), 'rb') as handle:
                X = pickle.load(handle)
        else:
            all_data, graph = read_BNrep(args)
            G = nx.DiGraph(graph)
            X = all_data['1000']['1']
    elif args.data_type == 'real':
        X, G = cdt.data.load_dataset('sachs')
        X = df_to_tensor(X)
    max_X_card = np.amax(X) + 1

    feat_train = torch.FloatTensor(X)
    feat_valid = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, G, max_X_card, X

def load_data(args, batch_size=1000, suffix='', debug = False):
    #  # configurations
    n, d = args.data_sample_size, args.data_variable_size
    graph_type, degree, sem_type, linear_type = args.graph_type, args.graph_degree, args.graph_sem_type, args.graph_linear_type
    x_dims = args.x_dims

    if args.data_type == 'synthetic':
        # generate data
        G = simulate_random_dag(d, degree, graph_type)
        X = simulate_sem(G, n, x_dims, sem_type, linear_type)

    elif args.data_type == 'discrete':
        # get benchmark discrete data
        if args.data_filename.endswith('.pkl'):
            with open(os.path.join(args.data_dir, args.data_filename), 'rb') as handle:
                X = pickle.load(handle)
        else:
            all_data, graph = read_BNrep(args)
            G = nx.DiGraph(graph)
            X = all_data['1000']['1']
    elif args.data_type == 'real':
        # get sachs
        X, G = cdt.data.load_dataset('sachs')
        X = df_to_tensor(X).unsqueeze(-1)
    feat_train = torch.Tensor(X)
    feat_valid = torch.Tensor(X)
    feat_test = torch.Tensor(X)
    # print(feat_train.size())

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, G


def load_numpy_data(args, data, adjacency):
    G = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph)
    data = torch.Tensor(data)
    data = TensorDataset(data, data)
    data_loader = DataLoader(data, batch_size=args.batch_size)
    return data_loader, G

# data generating functions

def simulate_random_dag(d: int,
                        degree: float,
                        graph_type: str,
                        w_range: tuple = (0.5, 2.0)) -> nx.DiGraph:
    """Simulate random DAG with some expected degree.

    Args:
        d: number of nodes
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)

    Returns:
        G: weighted DAG
    """
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    elif graph_type == 'barabasi-albert':
        m = int(round(degree / 2))
        B = np.zeros([d, d])
        bag = [0]
        for ii in range(1, d):
            dest = np.random.choice(bag, size=m)
            for jj in dest:
                B[ii, jj] = 1
            bag.append(ii)
            bag.extend(dest)
    elif graph_type == 'full':  # ignore degree, only for experimental use
        B = np.tril(np.ones([d, d]), k=-1)
    else:
        raise ValueError('unknown graph type')
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = 1*np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G


def simulate_sem(G: nx.DiGraph,
                 n: int, x_dims: int,
                 sem_type: str,
                 linear_type: str,
                 noise_scale: float = 1.0) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        X: [n,d] sample matrix
    """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d, x_dims])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        if linear_type == 'linear':
            eta = X[:, parents, 0].dot(W[parents, j])
        elif linear_type == 'nonlinear_1':
            eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
        elif linear_type == 'nonlinear_2':
            eta = (X[:, parents, 0]+0.5).dot(W[parents, j])
        else:
            raise ValueError('unknown linear data type')

        if sem_type == 'linear-gauss':
            if linear_type == 'linear':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_1':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_2':
                X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale, size=n)
        elif sem_type == 'linear-exp':
            X[:, j, 0] = eta + np.random.exponential(scale=noise_scale, size=n)
        elif sem_type == 'linear-gumbel':
            X[:, j, 0] = eta + np.random.gumbel(scale=noise_scale, size=n)
        else:
            raise ValueError('unknown sem type')
    if x_dims > 1 :
        for i in range(x_dims-1):
            X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
        X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
    return X


def simulate_population_sample(W: np.ndarray,
                               Omega: np.ndarray) -> np.ndarray:
    """Simulate data matrix X that matches population least squares.

    Args:
        W: [d,d] adjacency matrix
        Omega: [d,d] noise covariance matrix

    Returns:
        X: [d,d] sample matrix
    """
    d = W.shape[0]
    X = np.sqrt(d) * slin.sqrtm(Omega).dot(np.linalg.pinv(np.eye(d) - W))
    return X

if __name__ == "__main__":
    # -----------data parameters ------
    # configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default= 'synthetic',
                        choices=['synthetic', 'discrete', 'real'],
                        help='choosing which experiment to do.')
    parser.add_argument('--data_filename', type=str, default= 'alarm',
                        help='data file name containing the discrete files.')
    parser.add_argument('--data_dir', type=str, default= 'data/',
                        help='data file name containing the discrete files.')
    parser.add_argument('--data_sample_size', type=int, default=5000,
                        help='the number of samples of data')
    parser.add_argument('--data_variable_size', type=int, default=100,
                        help='the number of variables in synthetic generated data')
    parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                        help='the type of DAG graph by generation method')
    parser.add_argument('--graph_degree', type=int, default=3,
                        help='the number of degree in generated DAG graph')
    parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
                        help='the structure equation model (SEM) parameter type')
    parser.add_argument('--graph_linear_type', type=str, default='nonlinear_2',
                        help='the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z')
    parser.add_argument('--edge-types', type=int, default=2,
                        help='The number of edge types to infer.')
    parser.add_argument('--x_dims', type=int, default=1, #changed here
                        help='The number of input dimensions: default 1.')
    parser.add_argument('--z_dims', type=int, default=1,
                        help='The number of latent variable dimensions: default the same as variable size.')

    args = parser.parse_args()
    load_data(args)