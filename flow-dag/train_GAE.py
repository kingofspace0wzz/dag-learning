import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import time
import argparse
import pickle
import os
import datetime
import copy
import cdt
import math

import numpy as np

np.set_printoptions(linewidth=200)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
import pandas as pd
from cdt.utils.R import RPackages, launch_R_script

# from .dag_optim import compute_constraint, compute_jacobian_avg, is_acyclic
# from .utils.metrics import edge_errors, shd as shd_metric
# from .utils.penalty import compute_penalty, compute_group_lasso_l2_penalty
# from .utils.save import dump, load, np_to_csv
# from .utils.topo_sort import  generate_complete_dag
# from .plot import plot_weighted_adjacency, plot_adjacency, plot_learning_curves, plot_learning_curves_retrain
from utils import *
from sorting_model import Sinkhorn_Net, BirkhoffPoly
from sinkhorn_ops import *
from model import *
from data import *
from learnables import LearnableModel_NonLinGaussANM

import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import BirkhoffPolytope
from geoopt.optim import RiemannianAdam

from tqdm import tqdm

parser = argparse.ArgumentParser()

# -----------data parameters ------
# configurations
parser.add_argument('--data_type', type=str, default= 'synthetic',
                    choices=['synthetic', 'discrete', 'real'],
                    help='choosing which experiment to do.')
parser.add_argument('--data_filename', type=str, default= 'alarm',
                    help='data file name containing the discrete files.')
parser.add_argument('--data_dir', type=str, default= 'data/',
                    help='data file name containing the discrete files.')
parser.add_argument('--data_sample_size', type=int, default=5000,
                    help='the number of samples of data')
parser.add_argument('--data_variable_size', type=int, default=10,
                    help='the number of variables in synthetic generated data')
parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                    help='the type of DAG graph by generation method')
parser.add_argument('--graph_degree', type=int, default=3,
                    help='the number of degree in generated DAG graph')
parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
                    help='the structure equation model (SEM) parameter type: [linear-gauss, linear_exp, linear-gumbel]')
parser.add_argument('--graph_linear_type', type=str, default='nonlinear_2',
                    help='the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--x_dims', type=int, default=1, #changed here
                    help='The number of input dimensions: default 1.')
parser.add_argument('--z_dims', type=int, default=1,
                    help='The number of latent variable dimensions: default the same as variable size.')
parser.add_argument('--sink_z_dim', type=int, default=32)
parser.add_argument('--dropout_prob', type=float, default=0.0)
parser.add_argument('--mode', type=str, default='gae',
                    help='method, [gae, gran]')

# -----------training hyperparameters
parser.add_argument('--optimizer', type = str, default = 'Adam',
                    help = 'the choice of optimizer used')
parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                    help = 'threshold for learned adjacency matrix binarization')
parser.add_argument('--tau_A', type = float, default=1.,
                    help='coefficient for L-1 norm of A.')
parser.add_argument('--lambda_A',  type = float, default= 1.,
                    help='coefficient for DAG constraint h(A).')
parser.add_argument('--c_A',  type = float, default= 1,
                    help='coefficient for absolute value h(A).')
parser.add_argument('--lambda_sink', type=float, default=1., 
                    help='coefficient for sinkhorn loss')
parser.add_argument('--use_A_connect_loss',  type = int, default= 0,
                    help='flag to use A connect loss')
parser.add_argument('--use_A_positiver_loss', type = int, default = 0,
                    help = 'flag to enforce A must have positive values')
parser.add_argument('--sparse_b', type=float, default=1.)
parser.add_argument('--ortho_b', type=float, default=1.)
parser.add_argument('--sparse_A', type=float, default=1.)

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default= 200,
                    help='Number of epochs to train.')
parser.add_argument('--pre_train', type=int, default=10000,
                    help='Number of pre train epochs')
parser.add_argument('--post_train', type=int, default=100,
                    help='Number of post train after fixing permutation P')
parser.add_argument('--batch-size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--k_max_iter', type = int, default = 300,
                    help ='the max iteration number for searching lambda and c')

parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp, or sem).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--no_lower', action='store_true', default=False)
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')


parser.add_argument('--h_tol', type=float, default = 1e-8,
                    help='the tolerance of error of h(A) to zero')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default= 1.0,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

EPSULON = 1e-08

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    # safe_name = save_folder.text.replace('/', '_')
    os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

# ================================================
# get data: experiments = {synthetic SEM, ALARM}
# ================================================
train_loader, valid_loader, test_loader, ground_truth_G = load_data(args, args.batch_size, args.suffix)

#===================================
# load modules
#===================================
# Generate off-diagonal interaction graph
off_diag = np.ones([args.data_variable_size, args.data_variable_size]) - np.eye(args.data_variable_size)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
rel_rec = torch.DoubleTensor(rel_rec)
rel_send = torch.DoubleTensor(rel_send)

# add adjacency matrix A
num_nodes = args.data_variable_size
# adj_A = np.zeros((num_nodes, num_nodes))
adj_A = np.random.standard_normal((num_nodes, num_nodes))
# mask_A = np.ones((num_nodes, num_nodes), dtype=np.float) - np.eye(num_nodes, dtype=np.float)
args.lower = not args.no_lower
if args.lower:
    # mask_A = np.triu(np.ones((num_nodes, num_nodes), dtype=np.float)) - np.eye(num_nodes, dtype=np.float)
    mask_A = np.ones((num_nodes, num_nodes), dtype=np.float) - np.eye(num_nodes, dtype=np.float)
else:
    mask_A = np.ones((num_nodes, num_nodes), dtype=np.float) - np.eye(num_nodes, dtype=np.float)
if args.mode == 'gran':
    model = LearnableModel_NonLinGaussANM(args.data_variable_size * args.x_dims, 3, args.encoder_hidden)
else:
    model = GAE(args.data_variable_size * args.x_dims, args.x_dims, args.encoder_hidden,
                int(args.z_dims), adj_A, mask_A,
                batch_size=args.batch_size,
                do_prob=args.encoder_dropout, factor=args.factor, lower=args.lower)

# Sinkhorn net
sinkhorn_net = Sinkhorn_Net(args.sink_z_dim, args.data_variable_size, args.dropout_prob)

# BirkhoffPolytope
birkhoff = BirkhoffPoly(num_nodes)

#===================================
# set up training parameters
#===================================
if args.optimizer  == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'LBFGS':
    optimizer = optim.LBFGS(model.parameters(),
                            lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(),
                            lr=args.lr)  
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# set up Riemannian Adam
# rie_optimizer = RiemannianAdam(birkhoff.parameters(), lr=args.lr)
rie_optimizer = optim.Adam(birkhoff.parameters(), lr=args.lr)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.DoubleTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = log_prior

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    model.cuda()
    birkhoff.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    # triu_indices = triu_indices.cuda()
    # tril_indices = tril_indices.cuda()

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

def pns_(model_adj, train_data, test_data, num_neighbors, thresh):
    """Preliminary neighborhood selection"""
    x_train, _ = train_data.sample(train_data.num_samples)
    x_test, _ = test_data.sample(test_data.num_samples)
    x = np.concatenate([x_train.detach().cpu().numpy(), x_test.detach().cpu().numpy()], 0)

    num_samples = x.shape[0]
    num_nodes = x.shape[1]
    print("PNS: num samples = {}, num nodes = {}".format(num_samples, num_nodes))
    for node in range(num_nodes):
        print("PNS: node " + str(node))
        x_other = np.copy(x)
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(x_other, x[:, node])
        selected_reg = SelectFromModel(reg, threshold="{}*mean".format(thresh), prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False).astype(np.float)

        model_adj[:, node] *= mask_selected

    return model_adj


def pns(model, train_data, test_data, num_neighbors, thresh, exp_path, metrics_callback, plotting_callback):
    # Prepare path for saving results
    save_path = os.path.join(exp_path, "pns")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if already computed
    if os.path.exists(os.path.join(save_path, "DAG.npy")):
        print("pns already computed. Loading result from disk.")
        return load(save_path, "model.pkl")

    model_adj = model.adjacency.detach().cpu().numpy()
    time0 = time.time()
    model_adj = pns_(model_adj, train_data, test_data, num_neighbors, thresh)

    with torch.no_grad():
        model.adjacency.copy_(torch.Tensor(model_adj))

    timing = time.time() - time0
    print("PNS took {}s".format(timing))

    # save
    dump(model, save_path, 'model')
    dump(timing, save_path, 'timing')
    np.save(os.path.join(save_path, "DAG"), model.adjacency.detach().cpu().numpy())

    # plot
    plot_adjacency(model.adjacency.detach().cpu().numpy(), train_data.adjacency.detach().cpu().numpy(), save_path)

    return model

def train_lower(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []
    sid_train = []
    sink_train = []

    model.train()
    scheduler.step()


    # update optimizer
    # optimizer, lr = update_optimizer(optimizer, args.lr, c_A)

    # for i in range(10000):
    #     rie_optimizer.zero_grad()
    #     loss = 10 * torch.sum(torch.abs(birkhoff.Matrix)) + 10 * ortho_loss(birkhoff.Matrix)
    #     loss.backward()
    #     rie_optimizer.step()

    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda().float(), relations.cuda().float()
        # data = data.squeeze()
        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()
        rie_optimizer.zero_grad()

        log_alpha = birkhoff.Matrix
        permute, log_alpha_w_noise = gumbel_sinkhorn(log_alpha, temp=args.temp)
        inv_soft_perms = inv_soft_pers_flattened(permute)
        permute = permute.squeeze()
        x_permute = torch.matmul(permute, data)

        # use BirkhoffPolyope to permute the input

        # if args.epoch <= args.post_train:
        # x_permute = birkhoff(data)
        # x_permute = x_permute.squeeze()
        # permute = birkhoff.Matrix
        # print(permute)
        # else:
        #     permute = hard_permutation(birkhoff.Matrix)
        #     x_permute = torch.matmul(permute, data)

        if args.lower:
            preds, origin_A, myA = model(data, rel_rec, rel_send)
        else:
            preds, origin_A, myA = model(data, rel_rec, rel_send)
        
        if torch.sum(preds != preds):
            print('nan error\n')

        variance = 0.

        A_permute = torch.matmul(permute, origin_A)

        #=================================
        # Computing the losses
        #=================================

        # reconstruction accuracy loss
        # Gaussian, Maximum Likelihood
        # P X = A^T P X + Z, X = P^T A^T P X + P^T Z 
        # loss_nll = nll_gaussian(torch.matmul(torch.inverse(permute), preds), data, variance)
        # loss = nll_gaussian(preds, data, variance)
        # least squares
        if args.lower:
            # loss = F.mse_loss(preds, x_permute) + F.mse_loss(torch.matmul(permute.t(), preds), data)
            loss = F.mse_loss(preds, data)
        else:
            loss = F.mse_loss(preds, data)
       
        # MSE loss:
        # loss = F.mse_loss(preds, x_permute)

        # sinkhorn loss:
        # sink_loss = F.mse_loss(data, torch.matmul(inv_soft_perms, x_permute))
        # loss += args.lambda_sink * sink_loss

        # birkhoff loss:
        # loss += args.sparse_b * torch.sum(torch.abs(birkhoff.Matrix))
        # loss += args.ortho_b * ortho_loss(birkhoff.Matrix)
        # loss += args.sparse_A * torch.sum(torch.abs(torch.matmul(torch.inverse(permute), torch.matmul(origin_A, permute))))

        # Make P^T A triangular
        # loss += torch.norm(A_permute.tril() - torch.zeros_like(A_permute), p='fro')
        loss += torch.norm(A_permute.tril() - torch.zeros_like(A_permute), p=1)

        # add A loss
        one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))
        # loss += sparse_loss

        # other loss term
        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        if args.no_lower:
            # compute h(A)
            h_A = _h_A(origin_A, args.data_variable_size)
            # loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A 

        #=========================
        # computing the losses finished
        #=========================
        loss.backward()
        optimizer.step()
        rie_optimizer.step()
        # if args.epoch <= args.post_train:
        #     rie_optimizer.step()

        # myA.data = stau(myA.data, args.tau_A*args.lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # first we need convert the graph before the permutation, A' = P^(-1) A P
        # this is because after the permutation, we have P X = A P X + Z, and therefore X = P^(-1) A P X + P^(-1) Z
        # origin_A = torch.matmul(inv_soft_perms, torch.matmul(origin_A, soft_perms_inf))
        # print(origin_A.size())
       
        # if batch_idx % 100 == 0:
        #     print(origin_A)
        # hard_permute = hard_permutation(permute)
        # Adj = torch.matmul(hard_permute.t(), torch.matmul(origin_A, hard_permute))
        # print(permute)
        # Adj = torch.matmul(torch.inverse(permute), torch.matmul(origin_A, permute))
        Adj = origin_A
        # if args.epoch > args.post_train:
        #     permute = hard_permutation(permute)
        # print(origin_A)
        # compute metrics
        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, sid, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))


        mse_train.append(F.mse_loss(preds, data).item())
        # sink_train.append(sink_loss.item())
        shd_trian.append(shd)
        sid_train.append(sid)

    # print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    sink_val = []

    print('Epoch: {:04d}'.format(epoch),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
          'sid_train: {:.10f}'.format(np.mean(sid_train)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(model.state_dict(), encoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
              'sid_train: {:.10f}'.format(np.mean(sid_train)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')


    return np.mean(mse_train), graph, origin_A, origin_A, permute

def train_gran(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []
    sid_train = []
    sink_train = []

    model.train()
    scheduler.step()


    # update optimizer
    # optimizer, lr = update_optimizer(optimizer, args.lr, c_A)

    # for i in range(10000):
    #     rie_optimizer.zero_grad()
    #     loss = 10 * torch.sum(torch.abs(birkhoff.Matrix)) + 10 * ortho_loss(birkhoff.Matrix)
    #     loss.backward()
    #     rie_optimizer.step()

    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda().float(), relations.cuda().float()
        # data = data.squeeze()
        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()
        rie_optimizer.zero_grad()

        # use BirkhoffPolyope to permute the input

        # if args.epoch <= args.post_train:
        x_permute = birkhoff(data)
        # x_permute = x_permute.squeeze()
        permute = birkhoff.Matrix
        # print(permute)
        # else:
        #     permute = hard_permutation(birkhoff.Matrix)
        #     x_permute = torch.matmul(permute, data)

        if args.lower:
            preds, origin_A, myA = model(x_permute, rel_rec, rel_send)
        else:
            weights, biases, extra_params = model.get_parameters(mode='wbx')
        #     preds = torch.matmul(weights, data) + biases
        # if torch.sum(preds != preds):
        #     print('nan error\n')

        variance = 0.

        #=================================
        # Computing the losses
        #=================================

        # reconstruction accuracy loss
        # Gaussian, Maximum Likelihood
        # P X = A^T P X + Z, X = P^T A^T P X + P^T Z 
        # loss_nll = nll_gaussian(torch.matmul(torch.inverse(permute), preds), data, variance)
        # loss = nll_gaussian(preds, data, variance)
        # least squares
        if args.lower:
            loss = F.mse_loss(preds, x_permute) + F.mse_loss(torch.matmul(permute.t(), preds), data)
        else:
            loss = -torch.mean(model.compute_log_likelihood(data.unsqueeze(-1), weights, biases, extra_params))
            origin_A = model.get_w_adj()
       
        # birkhoff loss:
        # loss += args.sparse_b * torch.sum(torch.abs(birkhoff.Matrix))
        # loss += args.ortho_b * ortho_loss(birkhoff.Matrix)
        # loss += args.sparse_A * torch.sum(torch.abs(torch.matmul(torch.inverse(permute), torch.matmul(origin_A, permute))))

        # add A loss
        one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
        # sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))
        sparse_loss = args.tau_A * torch.norm(origin_A, p=1)

        # other loss term
        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        if args.no_lower:
            # compute h(A)
            h_A = _h_A(origin_A, args.data_variable_size)
            # loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A 

        #=========================
        # computing the losses finished
        #=========================
        loss.backward()
        loss = optimizer.step()
        if args.epoch <= args.post_train:
            rie_optimizer.step()

        # myA.data = stau(myA.data, args.tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # first we need convert the graph before the permutation, A' = P^(-1) A P
        # this is because after the permutation, we have P X = A P X + Z, and therefore X = P^(-1) A P X + P^(-1) Z
        # origin_A = torch.matmul(inv_soft_perms, torch.matmul(origin_A, soft_perms_inf))
        # print(origin_A.size())
       
        # if batch_idx % 100 == 0:
        #     print(origin_A)
        # hard_permute = hard_permutation(permute)
        # Adj = torch.matmul(hard_permute.t(), torch.matmul(origin_A, hard_permute))
        # print(permute)
        Adj = torch.matmul(torch.inverse(permute), torch.matmul(origin_A, permute))
        # if args.epoch > args.post_train:
        #     permute = hard_permutation(permute)
        # print(origin_A)
        # compute metrics
        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, sid, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))


        mse_train.append(loss.item())
        # sink_train.append(sink_loss.item())
        shd_trian.append(shd)
        sid_train.append(sid)

    # print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    sink_val = []

    print('Epoch: {:04d}'.format(epoch),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
          'sid_train: {:.10f}'.format(np.mean(sid_train)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(model.state_dict(), encoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
              'sid_train: {:.10f}'.format(np.mean(sid_train)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')


    return np.mean(mse_train), graph, origin_A, Adj, permute

def run():
    t_total = time.time()
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = args.c_A
    lambda_A = args.lambda_A
    h_A_new = torch.tensor(1.)
    h_tol = args.h_tol
    k_max_iter = int(args.k_max_iter)
    h_A_old = np.inf
    # if args.lower:
    #     for i in tqdm(range(args.pre_train)):
    #         rie_optimizer.zero_grad()
    #         loss = 1000000 * torch.sum(torch.abs(birkhoff.Matrix)) + 1000000 * ortho_loss(birkhoff.Matrix)
    #         loss.backward()
    #         rie_optimizer.step()
    args.epoch = 0
    try:
        if args.lower:
            for epoch in range(args.epochs):
                if args.mode == 'gran':
                    MSE_loss, graph, origin_A, Adj, permute = train_gran(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer)
                else:
                    MSE_loss, graph, origin_A, Adj, permute = train_lower(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer)
        
                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph
                if epoch % 20 == 0:
                    print('Permutation:\n {}'.format(permute))
                    print('origin A:\n {}'.format(origin_A))
                    print('I:\n{}'.format(torch.matmul(permute.t(), permute)))
                args.epoch += 1

            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
        else:
            for step_k in range(k_max_iter):
                while c_A < 1e+20:
                    for epoch in range(args.epochs):
                        if args.mode == 'gran':
                            MSE_loss, graph, origin_A, Adj, permute = train_gran(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer)
                        else:
                            MSE_loss, graph, origin_A, Adj, permute = train_lower(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer)
                
                        if MSE_loss < best_MSE_loss:
                            best_MSE_loss = MSE_loss
                            best_epoch = epoch
                            best_MSE_graph = graph
                        if epoch % 20 == 0:
                            print('Permutation:\n {}'.format(permute))
                            print('origin A:\n {}'.format(origin_A))
                            print('I:\n{}'.format(torch.matmul(permute.t(), permute)))
                        args.epoch += 1


                    print("Optimization Finished!")
                    print("Best Epoch: {:04d}".format(best_epoch))

                    # update parameters
                    A_new = origin_A.data.clone()
                    h_A_new = _h_A(A_new, args.data_variable_size)
                    if h_A_new.item() > 0.25 * h_A_old:
                        c_A*=10
                    else:
                        break

                    # update parameters
                    # h_A, adj_A are computed in loss anyway, so no need to store
                h_A_old = h_A_new.item()
                lambda_A += c_A * h_A_new.item()

                if h_A_new.item() <= h_tol:
                    break


        if args.save_folder:
            print("Best Epoch: {:04d}".format(best_epoch), file=log)
            log.flush()

        # test()

        print ('MSE graph:\n {}'.format(best_MSE_graph))
        print('ground truth:\n{}'.format(nx.to_numpy_array(ground_truth_G)))
        fdr, tpr, fpr, shd, sid, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
        print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


    except KeyboardInterrupt:
        # print the best anway
       
        print(best_MSE_graph)
        print(nx.to_numpy_array(ground_truth_G))
        fdr, tpr, fpr, shd, sid, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
        print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

run()