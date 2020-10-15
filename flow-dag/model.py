import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearSEMEncoder(nn.Module):
    """SEM encoder module."""
    def __init__(self, n_in, n_hid, n_out, adj_A, mask_A, batch_size, do_prob=0., factor=True, tol = 0.1, lower=True):
        super(LinearSEMEncoder, self).__init__()

        self.factor = factor
        self.adj_A = nn.Parameter(torch.from_numpy(adj_A).float(), requires_grad = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.lower = lower
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).float())
        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.mask_A = torch.Tensor(mask_A)
        
    def init_weights(self):
        nn.init.xavier_normal(self.adj_A.data)

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        adj_A1 = torch.sinh(3.*self.adj_A*self.mask_A)
        # if self.lower:
        #     self.adj_A1 = self.adj_A1.tril() - torch.eye(self.adj_A1.size(0)).float() * self.adj_A1.diag()
        # adj_A = I-A^T, adj_A_inv = (I-A^T)^(-1)
        adj_A = torch.eye(adj_A1.size(0)).float() - adj_A1.t()
        adj_A_inv = torch.inverse(adj_A)

        meanF = torch.matmul(adj_A_inv, torch.mean(torch.matmul(adj_A, inputs), 0))
        logits = torch.matmul(adj_A, inputs-meanF)

        return inputs-meanF, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

class LinearSEMDecoder(nn.Module):
    """SEM decoder module."""
    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(LinearSEMDecoder, self).__init__()

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = torch.inverse(torch.eye(origin_A.size(0)).float() - origin_A.t())
        mat_z = torch.matmul(adj_A_new1, input_z + Wa)
        out = mat_z

        return mat_z, out-Wa, adj_A_tilt

class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, mask_A, batch_size, do_prob=0., factor=True, tol = 0.1, lower=True):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(torch.from_numpy(adj_A).float(), requires_grad=True) # TODO: add masking for PNS
        self.mask_A = torch.Tensor(mask_A)
        self.factor = factor
        self.lower = lower
        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).float())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs, rel_rec, rel_send, permute=None):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A*self.mask_A)
        # if self.lower:
        #     self.adj_A1 = self.adj_A1.tril() - torch.eye(self.adj_A1.size(0)).float() * self.adj_A1.diag()
        # adj_Aforz = I-A^T
        adj_Aforz = torch.eye(adj_A1.size(0)) - adj_A1.t()

        adj_A = torch.eye(adj_A1.size()[0]).float()
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = torch.inverse(torch.eye(origin_A.size(0)) - origin_A.t())
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt

class GAE(nn.Module):
    """Some Information about GAE"""
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, mask_A, batch_size, do_prob=0., factor=True, tol = 0.1, lower=True):
        super(GAE, self).__init__()
        # self.adj_A = nn.Parameter(torch.from_numpy(adj_A).float(), requires_grad=True)
        self.adj_A = nn.Parameter(torch.zeros(n_in, n_in).float(), requires_grad=True)
        self.mask_A = torch.Tensor(mask_A)
        # self.fc1 = nn.Linear(n_in, n_in, bias = True)
        # self.fc2 = nn.Linear(n_in, n_in, bias = True)
        # self.fc3 = nn.Linear(n_in, n_in, bias = True)
        # self.fc4 = nn.Linear(n_in, n_in, bias = True)

        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_in, bias = True)
        self.fc3 = nn.Linear(n_in, n_hid, bias = True)
        self.fc4 = nn.Linear(n_hid, n_xdims, bias = True)
        self.lower = lower

    def forward(self, inputs, rel_rec, rel_send):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        # adj_A1 = self.adj_A
        # H = self.fc2(F.relu(self.fc1(inputs.squeeze()))).unsqueeze(-1)    # batch x d x 1
        # H2 = torch.matmul(adj_A1, H).squeeze() # batch x 10
        # x = self.fc4(F.relu(self.fc3(H2))).unsqueeze(-1)

        H = self.fc2(F.relu(self.fc1(inputs)))
        H2 = torch.matmul(adj_A1, H)
        x = self.fc4(F.relu(self.fc3(H2)))
        return x, adj_A1, self.adj_A

class FlowSEM(nn.Module):
    """Some Information about FlowSEMEncoder"""
    def __init__(self, A):
        super(FlowSEM, self).__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=True)
        self.flow = []
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        
        self.A = self.A.tril()
        adj_A = torch.inverse(torch.eye(A.size(0)) - self.A.t())
        out = self.flow(torch.matmul(adj_A, x))
        return x

