import cdt
from cdt.metrics import SID
from numpy.random import randint
import numpy as np
import torch
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import BirkhoffPolytope
from geoopt.optim import RiemannianAdam
from utils import hard_permutation
from geoopt.tensor import ManifoldTensor
from sinkhorn_ops import *
from sorting_model import BirkhoffPoly
 
birk = BirkhoffPoly(5)
x = torch.randn(2, 5, 1)
# p = hungarian(birk.Matrix)

# use sinkhorn gumbel to permute the input
log_alpha = birk.Matrix
soft_perms_inf, log_alpha_w_noise = gumbel_sinkhorn(log_alpha.unsqueeze(0), temp=0.1)
inv_soft_perms = inv_soft_pers_flattened(soft_perms_inf)

print(soft_perms_inf)
sums = soft_perms_inf.sum(0)