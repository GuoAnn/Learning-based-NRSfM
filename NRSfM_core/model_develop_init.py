import torch
import torch.nn as nn


class Fully_connection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Fully_connection, self).__init__()
        self.input_dim = input_dim  # dimension of the input state
        self.linear1 = nn.Linear(in_features=input_dim, out_features=input_dim, bias=True) #2 means the partial derivative terms
        self.linear2 = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)  # 2 means the partial derivative terms
        self.linear3 = nn.Linear(in_features=output_dim, out_features=output_dim, bias=True)  # 2 means the partial derivative terms
        self.activation = nn.PReLU(init=0.25)

    def forward(self, Input):
        # 9 layers fully connection layer
        # 1-th layer
        for i in range(5):
            out = self.linear1(Input)# Size of linear is (ln+s)*3, so we have V*(ln+s)*(ln+s)*3=V*3
            out = self.activation(out)
            out1 = self.linear1(out)# Size of linear is (ln+s)*3, so we have V*(ln+s)*(ln+s)*3=V*3
            out1 = self.activation(out1)
            out2 = self.linear1(out1)# Size of linear is (ln+s)*3, so we have V*(ln+s)*(ln+s)*3=V*3
            Input = self.activation(out2+Input+out)


        # 2-th layer
        out3 = self.linear2(Input)
        out3 = self.activation(out3)
        # 3-th layer

        for i in range(5):
            out4 = self.linear3(out3)
            out4 = self.activation(out4)
            out5 = self.linear3(out4)
            out5 = self.activation(out5)
            out6 = self.linear3(out5)
            out3 = self.activation(out6+out3+out4)

        out_all = out3

        return out_all
