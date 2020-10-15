import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import BirkhoffPolytope
from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldTensor
from utils import ortho_loss
import numpy as np

def my_sample_permutations(n_permutations, n_objects):
    """Samples a batch permutations from the uniform distribution.

    Returns a sample of n_permutations permutations of n_objects indices.
    Permutations are assumed to be represented as lists of integers
    (see 'listperm2matperm' and 'matperm2listperm' for conversion to alternative
    matricial representation). It does so by sampling from a continuous
    distribution and then ranking the elements. By symmetry, the resulting
    distribution over permutations must be uniform.

    Args:
    n_permutations: An int, the number of permutations to sample.
    n_objects: An int, the number of elements in the permutation.
      the embedding sources.

    Returns:
    A 2D integer tensor with shape [n_permutations, n_objects], where each
      row is a permutation of range(n_objects)

    """
    random_pre_perm = torch.empty(n_permutations, n_objects).uniform_(0, 1)
    _, permutations = torch.topk(random_pre_perm, k = n_objects)
    return permutations

def my_listperm2matperm(listperm):
    """Converts a batch of permutations to its matricial form.

    Args:
    listperm: 2D tensor of permutations of shape [batch_size, n_objects] so that
      listperm[n] is a permutation of range(n_objects).

    Returns:
    a 3D tensor of permutations matperm of
      shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
      permutation of the identity matrix, with matperm[n, i, listperm[n,i]] = 1
    """
    n_objects = listperm.size()[0]
    eye = np.eye(n_objects)[listperm]
    eye= torch.tensor(eye, dtype=torch.int32)
    return eye

class Sinkhorn_Net(nn.Module):

    def __init__(self, latent_dim, output_dim, dropout_prob):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        in_flattened_vector: input flattened vector
        latent_dim: number of neurons in latent layer
        output_dim: dimension of log alpha square matrix
        """
        super(Sinkhorn_Net, self).__init__()
        self.output_dim = output_dim

        # net: output of the first neural network that connects numbers to a
        # 'latent' representation.
        # activation_fn: ReLU is default hence it is specified here
        # dropout p – probability of an element to be zeroed
        self.linear1 = nn.Linear(1, latent_dim)
        self.relu1 = nn.ReLU()
        self.d1 = nn.Dropout(p = dropout_prob)
        # now those latent representation are connected to rows of the matrix
        # log_alpha.
        self.linear2 = nn.Linear(latent_dim, output_dim)
        self.d2 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        # each number is processed with the same network, so data is reshaped
        # so that numbers occupy the 'batch' position.
        x = x.view(-1, 1)
        # activation_fn: ReLU
        x = self.d1(self.relu1(self.linear1(x)))
        # no activation function is enabled
        x = self.d2(self.linear2(x))
        #reshape to cubic for sinkhorn operation
        x = x.reshape(-1, self.output_dim, self.output_dim)
        return x

class Sinkhorn_A(nn.Module):
    """Some Information about Sinkhorn_A
    Parameterize S as S = g(A)
    """
    def __init__(self, latent_dim, output_dim, dropout_prob):
        super(Sinkhorn_A, self).__init__()
        self.output_dim = output_dim

        # net: output of the first neural network that connects numbers to a
        # 'latent' representation.
        # activation_fn: ReLU is default hence it is specified here
        # dropout p – probability of an element to be zeroed
        self.linear1 = nn.Linear(1, latent_dim)
        self.relu1 = nn.ReLU()
        self.d1 = nn.Dropout(p = dropout_prob)
        # now those latent representation are connected to rows of the matrix
        # log_alpha.
        self.linear2 = nn.Linear(latent_dim, output_dim)
        self.d2 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        input: x, a [n,n] matrix
        """
        # each number is processed with the same network, so data is reshaped
        # so that numbers occupy the 'batch' position.
        x = x.view(-1, 1)
        # activation_fn: ReLU
        x = self.d1(self.relu1(self.linear1(x)))
        # no activation function is enabled
        x = self.d2(self.linear2(x))
        #reshape to cubic for sinkhorn operation
        x = x.reshape(-1, self.output_dim, self.output_dim)
        return x

class BirkhoffPoly(nn.Module):
    """Some Information about Birhoff"""
    def __init__(self, n):
        super(BirkhoffPoly, self).__init__()
        self.n = n
        self.manifold = BirkhoffPolytope()

        # Initialize permutation
        # per = my_sample_permutations(1, n)
        # perm = torch.randperm(n)
        # self.initial_permutation = my_listperm2matperm(perm).squeeze(0).float() + torch.normal(mean=torch.zeros(n, n).float(), std=torch.ones(n, n)/10)
        # self.initial_permutation = my_listperm2matperm(per).squeeze(0).float() + torch.normal(mean=torch.zeros(n, n).float(), std=torch.ones(n, n)/10)
        # self.initial_permutation = ManifoldTensor(self.manifold.projx(self.initial_permutation), manifold=BirkhoffPolytope())
        self.Matrix = ManifoldParameter(
                data=self.manifold.random(n, n),
                # data = self.initial_permutation,
                manifold=self.manifold
            )
        

    def forward(self, x=None):
        # out = P^T x
        out = torch.matmul(self.Matrix.unsqueeze(0), x)
        return out

if __name__ == "__main__":
    birh = BirkhoffPoly(10).cuda()
    rie_adam = RiemannianAdam(birh.parameters())
    inputs = torch.randn(2,10, 1)
    inputs_permute = inputs.clone()
    inputs_permute[:, 1], inputs_permute[:, 2] = inputs[:, 2].clone(), inputs[:, 1].clone()
    print(inputs)
    print(inputs_permute)
    print('before {}'.format(birh.Matrix))
    print(torch.inverse(birh.Matrix))
    for iter in range(20):
        rie_adam.zero_grad()
        loss = 10*ortho_loss(birh.Matrix) + 10*torch.sum(torch.abs(birh.Matrix))
        print(loss)
        loss.backward()
        rie_adam.step()
    print('after {}'.format(birh.Matrix))
    for iter in range(10): 
        rie_adam.zero_grad()
        out = birh(inputs_permute)
        # loss = torch.zeros(1)
        # for param in birh.parameters():
        #     if isinstance(param, geoopt.tensor.ManifoldParameter):
        #         loss = ortho_loss(birh.Matrix)
        loss = F.mse_loss(out, inputs) + ortho_loss(birh.Matrix) + 0*torch.sum(torch.abs(birh.Matrix))
        loss.backward()
        rie_adam.step()
        if iter % 100 == 0:
            print(loss) 
    print(birh.Matrix)
    print(birh.Matrix.sum(0))
    print(torch.matmul(birh.Matrix.t(), birh.Matrix))
