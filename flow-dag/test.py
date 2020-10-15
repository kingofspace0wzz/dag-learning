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
 
birk = BirkhoffPolytope()
man = torch.Tensor([[1,0,0],[0,0,1],[0,1,0]])
a = ManifoldTensor(birk.projx(man), manifold=BirkhoffPolytope())
print(a)