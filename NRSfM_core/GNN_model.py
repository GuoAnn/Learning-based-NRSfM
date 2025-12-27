import torch
import torch.nn as nn
import numpy as np

# Xi means the \Xi in eq.13 An,u matrix
class Xi(nn.Module):
    def __init__(self, ln, s):
        super(Xi, self).__init__()
        self.ln = ln   # dimension of the state
        self.s = s     # Number of the node
        self.fc1 = torch.nn.Linear(in_features=2 * ln, out_features=2 * ln, bias=True) #2*ln include the feature of the node itself and the feature of the neib node
        self.fc1_add = torch.nn.Linear(in_features=2 * ln, out_features=2 * ln, bias=True)  # 2*ln include the feature of the node itself and the feature of the neib node
        self.fc2 = torch.nn.Linear(in_features=2 * ln, out_features=s ** 2, bias=True)
        self.fc3 = torch.nn.Linear(in_features=s ** 2, out_features=s ** 2, bias=True)
        self.fc3_add = torch.nn.Linear(in_features=s ** 2, out_features=s ** 2, bias=True)
        self.activation = torch.nn.PReLU()
        self.Norm1d_1 = nn.BatchNorm1d(2 * ln)
        self.Norm1d_2 = nn.BatchNorm1d(s ** 2)

    def forward(self, X):
        bs = X.size()[0]
        fc1_out=X
        for i in range(5):
            #fc1_out = self.activation(self.fc1(fc1_out))#[ 1002, 10000] loss: 0.288 2.246
            #fc1_out = self.activation(self.fc1(fc1_out))#[ 1002, 10000] loss: 0.085 0.820
            fc1_out = self.activation(self.fc1_add(fc1_out)+X)#[ 1002, 10000] loss: 0.076 0.471 ([ 1002, 10000] loss: 0.062 0.465 only)
            #fc1_out = self.Norm1d_1(self.activation(self.fc1_add(fc1_out) + X))#[  999, 10000] loss: 0.074 0.498

        fc2_out = self.activation(self.fc2(fc1_out))

        fc3_out= fc2_out
        for i in range(5):#[ 1002, 10000] loss: 0.075 0.664  10 iterations  #[ 1002, 10000] loss: 0.109 0.653 3 iterations
            fc3_out = self.activation(self.fc3(fc3_out)) #[ 1002, 10000] loss: 0.567 6.592
            fc3_out = self.activation(self.fc3_add(fc3_out)+fc2_out) #[ 1002, 10000] loss: 0.252 2.255
            #fc3_out = self.activation(self.fc3_add(fc3_out) + fc2_out + self.fc2(X))
        return fc3_out.view(bs, self.s, self.s)


class Rou(nn.Module):
    def __init__(self, ln, s):
        super(Rou, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=ln, out_features=ln, bias=True)
        self.fc1_add = torch.nn.Linear(in_features=ln, out_features=ln, bias=True)
        self.fc2 = torch.nn.Linear(in_features=ln, out_features=s, bias=True)
        self.fc3 = torch.nn.Linear(in_features=s, out_features=s, bias=True)
        self.fc3_add = torch.nn.Linear(in_features=s, out_features=s, bias=True)
        self.activation = torch.nn.PReLU()
    def forward(self, X):
        fc1_out = X
        for i in range(5):
            #fc1_out = self.activation(self.fc1(X))
            fc1_out = self.activation(self.fc1_add(fc1_out)+X) #[ 1002, 10000] loss: 0.067 0.527

        fc2_out = self.activation(self.fc2(fc1_out))
        fc3_out = fc2_out

        for i in range(5):
            #fc3_out = self.activation(self.fc3(fc2_out))
            fc3_out = self.activation(self.fc3_add(fc3_out)+fc2_out)
        return fc3_out

class Hw(nn.Module):
    def __init__(self, ln, s, degree, mu=0.8):
        super(Hw, self).__init__()
        self.ln = ln   # dimension of the state
        self.s = s     # Number of the node
        self.Xi = Xi(ln, s)
        self.Rou = Rou(ln, s)
        self.mu = mu
        self.dg_list = degree

    def forward(self, X, H):
        A = self.mu/self.s/self.dg_list*self.Xi(X)
        b = self.Rou(torch.chunk(X, chunks=2, dim=1)[0])
        return torch.matmul(A, torch.unsqueeze(H, 2))+torch.unsqueeze(b,2)

class AggrSum(nn.Module):
    def __init__(self, node_num, device):
        super(AggrSum, self).__init__()
        self.V = node_num
        self.device = device

    def forward(self, H, X_node):
        mask = torch.stack([X_node] * self.V, 0)
        mask = mask.float() - torch.unsqueeze(torch.arange(0, self.V).float(), 1).to(self.device)
        mask = (mask == 0).float()
        return torch.mm(mask, torch.squeeze(H))


class Non_LinearGNN(nn.Module):
    def __init__(self, node_num, device, feat_dim, stat_dim, iteration, degree):
        super(Non_LinearGNN, self).__init__()
        self.node_num=node_num  #All nodes
        self.feat_dim=feat_dim
        self.stat_dim=stat_dim  #Latent space (state vector corresponding to the nodes
        self.iteration=iteration
        self.degree=degree
        self.device=device
        self.Hw = Hw(feat_dim, stat_dim, degree)
        self.node_states = nn.Parameter(torch.zeros((self.node_num, self.stat_dim), dtype=torch.float32), requires_grad = True)#Initial state vector
        #Random initialization for node_feature table x-axis: all nodes, y-axis: feature(label) number V*ln
        self.linear1 = nn.Linear(feat_dim + stat_dim, feat_dim + stat_dim) #2 means the partial derivative terms
        self.linear1_add = nn.Linear(feat_dim + stat_dim, feat_dim + stat_dim)  # 2 means the partial derivative terms
        self.linear2 = nn.Linear(feat_dim + stat_dim, 2)  # 2 means the partial derivative terms
        self.linear3 = nn.Linear(2, 2)  # 2 means the partial derivative terms
        self.linear3_add = nn.Linear(2, 2)  # 2 means the partial derivative terms
        self.activation = nn.PReLU(init=0.25)
        self.Aggr = AggrSum(node_num, self.device)
        self.Norm1d_1 = nn.BatchNorm1d(feat_dim + stat_dim)
        self.Norm1d_2 = nn.BatchNorm1d(2)

    def forward(self, X_Node, X_Neis, feature_Matrix):
        #Pick out the lines corresponding to the node X_node in the list size: N*ln
        node_embeds = torch.index_select(input=torch.transpose(feature_Matrix, 0, 1), dim=0, index=X_Node)
        #Pick out the lines corresponding to the neigbour node X_Neis of the node X_node in the list N*ln
        neis_embeds = torch.index_select(input=torch.transpose(feature_Matrix, 0, 1), dim=0, index=X_Neis)
        #Combine two tensors based on the list (edge) x-axis is the node in the list list。 y-axis is two classes of features corresponding two nodes
        #Size: N*(ln+ln)
        X = torch.cat((node_embeds, neis_embeds), 1)
        #Initial state
        H = self.node_states


        for t in range(self.iteration):
            H = torch.index_select(H, 0, X_Neis)
            H = self.Hw(X, H)
            H = self.Aggr(H, X_Node)
        #gw function mapping from the stable point H (state) and their features (labels) self.node_features.data
        out = torch.cat((torch.transpose(feature_Matrix,0,1), H), 1)# V*(ln+s)
        # 3 layers fully connection layer
        # 1-th layer
        out1 = out
        for i in range(10):
            out1 = self.linear1(out1)# Size of linear is (ln+s)*3, so we have V*(ln+s)*(ln+s)*3=V*3
            out1 = self.activation(out1)
            out1 = self.linear1_add(out1)# Size of linear is (ln+s)*3, so we have V*(ln+s)*(ln+s)*3=V*3
            out1 = self.Norm1d_1(self.activation(out1+out))

        # 2-th layer
        out1 = self.linear2(out1)
        out2 = self.activation(out1)
        out3 = out2
        # 3-th layer
        for i in range(10):
            out3 = self.linear3(out3)
            out3 = self.activation(out3)
            out3 = self.linear3_add(out3)
            out3 = self.activation(out3+out2)



        out3_1 = out3[:, 0]
        out3_2 = out3[:, 1]
        out3_all = torch.cat((out3_1, out3_2), 0)

        return out3_all #The output means the problaity of the class for every node
        #Its size is (V,3). Every row means the node and the probalility of it belongs to the class