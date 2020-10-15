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


def count_accuracy(G_true: nx.DiGraph,
                   G: nx.DiGraph,
                   G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # sid = float(cdt.metrics.SID(target=G_true, pred=G))
    sid = 0 # TODO: resolve the cdt.metrics.SID bug
    return fdr, tpr, fpr, shd, sid, pred_size



#===============================
# KL and Likelihood
def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - torch.log(log_prior + eps))
    return kl_div.sum() / (num_atoms)

def kl_gaussian(preds, zsize):
    predsnew = preds.squeeze(1)
    mu = predsnew[:,0:zsize]
    log_sigma = predsnew[:,zsize:2*zsize]
    kl_div = torch.exp(2*log_sigma) - 2*log_sigma + mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)) - zsize)*0.5

def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5

def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def nll_catogrical(preds, target, add_const = False):
    '''compute the loglikelihood of discrete variables
    '''
    # loss = nn.CrossEntropyLoss()

    total_loss = 0
    for node_size in range(preds.size(1)):
        total_loss += - (torch.log(preds[:,node_size, target[:, node_size].long()]) * target[:, node_size]).mean()

    return total_loss

def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


# DATA
def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('_graph' + extension))


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

    feat_train = torch.Tensor(X)
    feat_valid = torch.Tensor(X)
    feat_test = torch.Tensor(X)

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


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    print(tril_indices)
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket

#Symmetrically normalize adjacency matrix.
def normalize_adj(adj):
    rowsum = torch.abs(torch.sum(adj,1))
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    myr = torch.matmul(torch.matmul(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)
    myr[isnan(myr)] = 0.
    return myr

def preprocess_adj(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() + (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized

def isnan(x):
    return x!=x

def my_normalize(z):
    znor = torch.zeros(z.size()).double()
    for i in range(z.size(0)):
        testnorm = torch.norm(z[i,:,:], dim=0)
        znor[i,:,:] = z[i,:,:]/testnorm
    znor[isnan(znor)] = 0.0
    return znor

def sparse_to_tuple(sparse_mx):
#    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def matrix_poly(matrix, d):
    x = torch.eye(d, device=matrix.device).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)


# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss +=  2 * tol - torch.sum(torch.abs(A[:,i])) - torch.sum(torch.abs(A[i,:])) + z * z
    return loss

# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):
    result = - A + z_positive * z_positive
    loss =  torch.sum(result)

    return loss


'''
COMPUTE SCORES FOR BN
'''
def compute_BiCScore(G, D):
    '''compute the bic score'''
    # score = gm.estimators.BicScore(self.data).score(self.model)
    origin_score = []
    num_var = G.shape[0]
    for i in range(num_var):
        parents = np.where(G[:,i] !=0)
        score_one = compute_local_BiCScore(D, i, parents)
        origin_score.append(score_one)

    score = sum(origin_score)

    return score


def compute_local_BiCScore(np_data, target, parents):
    # use dictionary
    sample_size = np_data.shape[0]
    var_size = np_data.shape[1]

    # build dictionary and populate
    count_d = dict()
    if len(parents) < 1:
        a = 1

    # unique_rows = np.unique(self.np_data, axis=0)
    # for data_ind in range(unique_rows.shape[0]):
    #     parent_combination = tuple(unique_rows[data_ind,:].reshape(1,-1)[0])
    #     count_d[parent_combination] = dict()
    #
    #     # build children
    #     self_value = tuple(self.np_data[data_ind, target].reshape(1,-1)[0])
    #     if parent_combination in count_d:
    #         if self_value in count_d[parent_combination]:
    #             count_d[parent_combination][self_value] += 1.0
    #         else:
    #             count_d[parent_combination][self_value] = 1.0
    #     else:
    #         count_d[parent_combination] = dict()
    #         count_d

    # slower implementation
    for data_ind in range(sample_size):
        parent_combination = tuple(np_data[data_ind, parents].reshape(1, -1)[0])
        self_value = tuple(np_data[data_ind, target].reshape(1, -1)[0])
        if parent_combination in count_d:
            if self_value in count_d[parent_combination]:
                count_d[parent_combination][self_value] += 1.0
            else:
                count_d[parent_combination][self_value] = 1.0
        else:
            count_d[parent_combination] = dict()
            count_d[parent_combination][self_value] = 1.0

    # compute likelihood
    loglik = 0.0
    # for data_ind in range(sample_size):
    # if len(parents) > 0:
    num_parent_state = np.prod(np.amax(np_data[:, parents], axis=0) + 1)
    # else:
    #    num_parent_state = 0
    num_self_state = np.amax(np_data[:, target], axis=0) + 1

    for parents_state in count_d:
        local_count = sum(count_d[parents_state].values())
        for self_state in count_d[parents_state]:
            loglik += count_d[parents_state][self_state] * (
                        math.log(count_d[parents_state][self_state] + 0.1) - math.log(local_count))

    # penality
    num_param = num_parent_state * (
                num_self_state - 1)  # count_faster(count_d) - len(count_d) - 1 # minus top level and minus one
    bic = loglik - 0.5 * math.log(sample_size) * num_param

    return bic

def hard_permutation(perm):
    # n = perm.size(0)
    # _, id = torch.max(perm, dim=1)
    perm = perm.detach().numpy()
    for i in range(perm.shape[1]):
        perm[i] = np.where(perm[i, :] < perm.max(1)[0], 0, 1)
    # perm = np.ones_like(perm) - perm.astype(float)
    return torch.from_numpy(perm)

#================
# Compute loss for Birkhoff
#================
def ortho_loss(A):
    n = A.size(0)
    return (torch.matmul(A.t(), A) - torch.eye(n, device=A.device).float()).abs().sum()

if __name__ == "__main__":
    a = torch.randn(5,5)
    print(a)
    perm = hard_permutation(a)
    print(perm)