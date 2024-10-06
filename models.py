import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import *
from torch.autograd import Variable
import math


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        neural_num = 200
        d_prob = 0.2
        self.linears = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(100, 1),
        )
        self.linears2 = nn.Sequential(
            nn.Linear(50, 100),#63175
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            #nn.Dropout(d_prob),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )
    def forward(self,x):
        x1 = self.linears(x)
        return x1

class MLP2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        neural_num = 200
        d_prob = 0.2
        self.linears = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(100, 1),
        )
        self.linears2 = nn.Sequential(
            nn.Linear(50, 100),#63175
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            #nn.Dropout(d_prob),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )
    def forward(self,x):
        x1 = self.linears(x)
        return x1

class MLP_addcov2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        neural_num = 200
        d_prob = 0.2
        self.linears = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(100, 10),
        )
        self.linears2 = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10)
        )

        self.linears3 = nn.Sequential(
            nn.Linear(12, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )

    def forward(self,x, cov2):
        x1 = self.linears(x)
        #x2 = self.linears2(cov2)
        x1_x2 = torch.cat([x1, cov2], dim=1) 
        x3 = self.linears3(x1_x2)
        return x3

class MLP_cov2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        neural_num = 200
        d_prob = 0
        self.linears = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(10, 1)
        )
    def forward(self,x):
        x1 = self.linears(x)
        return x1

class MLP_PCA10(nn.Module):
    def __init__(self):
        super().__init__()
        neural_num = 200
        d_prob = 0.1
        self.linears = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(d_prob),
            nn.Linear(100, 10),
            nn.ReLU(inplace=True),
            #nn.Dropout(d_prob),
            nn.Linear(10, 1),
        )
    def forward(self,x):
        x1 = self.linears(x)
        return x1

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(10, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(17152, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim*2148*2, 128)
        self.fc2 = nn.Linear(128, output_dim)

        self.linears = nn.Sequential(
            nn.Linear(10*2148, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        # out = x.view(x.shape[0], -1)
        # out = self.linears(out)

        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out.contiguous().view(out.shape[0], -1)
        out = self.fc2(self.fc1(out))
        return out

neurons=714
dropout=0.1
primary_capslen=8
digital_capslen=16
ks=5
stride=2
filters=32
num_iterations=3 #danymic routing iterations
top_k=989 #89222
act=F.relu
device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class ConvCaps2D(nn.Module):
    def __init__(self):
        super(ConvCaps2D, self).__init__()
        self.capsules = nn.ModuleList([nn.Conv1d(in_channels=32, out_channels = primary_capslen,
                                                 kernel_size=ks, stride=stride) for _ in range(filters)])

    def squash(self, tensor, dim=-1):
        norm = (tensor**2).sum(dim=dim, keepdim = True) # norm.size() is (None, 1152, 1)
        scale = norm / (1 + norm) # scale.size()  is (None, 1152, 1)
        return scale*tensor / torch.sqrt(norm)

    def forward(self, x):
        outputs = [capsule(x).view(x.size(0), primary_capslen, -1) for capsule in self.capsules] # 32 list of (None, 1, 8, 36)
        outputs = torch.cat(outputs, dim = 2).permute(0, 2, 1)  # outputs.size() is (None, 1152, 8)
        return self.squash(outputs)


class Caps1D(nn.Module):
    def __init__(self):
        super(Caps1D, self).__init__()
        self.num_iterations = num_iterations
        self.num_caps = 2 # equals to class number
        self.num_routes= (int((neurons-ks)/stride)+1)*filters
        self.in_channels=primary_capslen
        self.out_channels=digital_capslen

        self.W = nn.Parameter(torch.randn(self.num_caps,self.num_routes, self.in_channels, self.out_channels)) # class,weight,len_capsule,capsule_layer
#         self.W = nn.Parameter(torch.randn(3, 3136, 8, 32)) # num_caps, num_routes, in_channels, out_channels

    def softmax(self, x, dim = 1):
        transposed_input = x.transpose(dim, len(x.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(x.size()) - 1)

    def squash(self, tensor, dim=-1):
        norm = (tensor**2).sum(dim=dim, keepdim = True) # norm.size() is (None, 1152, 1)
        scale = norm / (1 + norm)
        return scale*tensor / torch.sqrt(norm)

    # Routing algorithm
    def forward(self, u):
        # u.size() is (None, 1152, 8)
        '''
        From documentation
        For example, if tensor1 is a j x 1 x n x m Tensor and tensor2 is a k x m x p Tensor,
        out will be an j x k x n x p Tensor.
        We need j = None, 1, n = 1152, k = 10, m = 8, p = 16
        '''

        u_ji = torch.matmul(u[:, None, :, None, :], self.W) # u_ji.size() is (None, num_caps, 1152, 1, digital_capslen)

        b = Variable(torch.zeros(u_ji.size())) # b.size() is (None, 10, 1152, 1, 16)
        b = b.to(device) # using gpu

        for i in range(self.num_iterations):
            c = self.softmax(b, dim=2)
            v = self.squash((c * u_ji).sum(dim=2, keepdim=True)) # v.size() is (None, 10, 1, 1, 16)
            if i != self.num_iterations - 1:
                delta_b = (u_ji * v).sum(dim=-1, keepdim=True)
                b = b + delta_b

        # Now we simply compute the length of the vectors and take the softmax to get probability.
        v = v.squeeze() # None, num_caps, digital_capslen
        f = v.view(v.size(0),  -1)
        
        classes = (v ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)

        y_pred = torch.argmax(classes, axis=1)
        one_hot = F.one_hot(y_pred, num_classes=2).unsqueeze(2).repeat(1, 1, 16)
        v1 = v * one_hot
        v1 = v1.squeeze() # None, num_caps, digital_capslen
        f1 = v1.view(v1.size(0),  -1)
        return classes, f

class Squash(nn.Module):
    def __init__(self, eps=1e-20):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        coef = 1 - 1 / (torch.exp(norm) + self.eps)
        unit = x / (norm + self.eps)
        return coef * unit

class RoutingCaps(nn.Module):
    def __init__(self, in_capsules, out_capsules):
        super(RoutingCaps, self).__init__()
        self.N0, self.D0 = in_capsules
        self.N1, self.D1 = out_capsules
        self.squash = Squash()

        # initialize routing parameters
        self.W = nn.Parameter(torch.Tensor(self.N1, self.N0, self.D0, self.D1))
        nn.init.kaiming_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(self.N1, self.N0, 1))

    def forward(self, x):
        ## prediction vectors
        # ji,kjiz->kjz = k and z broadcast, then ji,ji->j = sum(a*b,axis=1)
        u = torch.einsum("...ji,kjiz->...kjz", x, self.W)  # (batch_size/B, N1, N0, D1)
        #print(x.shape, self.W.shape, u.shape)
        ## coupling coefficients
        # ij,kj->i = ij,kj->k = sum(matmul(a,a.T),axis=0) != ij,ij->i
        c = torch.einsum("...ij,...kj->...i", u, u)  # (B, N1, N0)
        c = c[..., None]  # (B, N1, N0, 1) for bias broadcasting
        c = c / torch.sqrt(torch.tensor(self.D1).float())  # stabilize
        c = torch.softmax(c, axis=1) + self.b

        ## new capsules
        s = torch.sum(u * c, dim=-2)  # (B, N1, D1)
        return self.squash(s)

class CapsMask(nn.Module):
    def __init__(self):
        super(CapsMask, self).__init__()

    def forward(self, x, y_true=None):
        if y_true is not None:  # training mode
            mask = y_true
        else:  # testing mode
            # convert list of maximum value's indices to one-hot tensor
            temp = torch.sqrt(torch.sum(x**2, dim=-1))
            mask = F.one_hot(torch.argmax(temp, dim=1), num_classes=temp.shape[1])
        
        masked = x * mask.unsqueeze(-1)
        return masked.view(x.shape[0], -1)  # reshape

class ReconstructionNet(nn.Module):
    def __init__(self, input_size=(2148, 10), num_classes=2, num_capsules=16):
        super(ReconstructionNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(in_features=num_capsules * num_classes, out_features=512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, np.prod(input_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_normal_(self.fc3.weight)  # glorot normal

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1, *self.input_size)  # reshape

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.num_features = in_channels+num_layers * growth_rate
        for i in range(num_layers):
            layer = self._create_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def _create_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        return layer

    def forward(self, x):
        features = [x]

        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)

        return torch.cat(features, dim=1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool1d(9, stride=5)
        )

    def forward(self, x):
        return self.layer(x)

class CapECG(nn.Module):
    def __init__(self, input_len, input_dim, hidden_dim, num_layers, output_dim):
        super(CapECG, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.maxpool = nn.MaxPool1d(hidden_dim)
        self.avgpool = nn.AvgPool1d(hidden_dim)
        self.Conv1d_2 = nn.Conv1d(in_channels=2, out_channels = 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

        self.Conv1d_1 = nn.Conv1d(in_channels=hidden_dim, out_channels = 32, kernel_size=9, stride=3)
        len1 = math.floor((input_len-9)/3)+1
        len2 = (math.floor((len1-5)/2)+1)*32
        self.lstm = nn.LSTM(32, 16, batch_first=True, bidirectional=True)
        self.ConvCaps2D = ConvCaps2D()
        self.routing_caps = RoutingCaps(in_capsules=(len2, 8), out_capsules=(2, 16))
        self.digitCaps = Caps1D()
        self.linears1 = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,1)
        )

        self.mask = CapsMask()
        self.generator = ReconstructionNet(input_size=(input_len, input_dim))
        self.num_layers = 4
        self.growth_rate = 32
        self.compression_rate = 0.3
        self.dense_block_num = 3
        self.dense_layer = nn.ModuleList()
        in_features = 10
        for i in range(self.dense_block_num):
            self.dense_layer.append(DenseBlock(in_features, self.growth_rate, self.num_layers))
            self.dense_layer.append(Transition(in_features + self.num_layers * self.growth_rate, int((in_features + self.num_layers * self.growth_rate)*self.compression_rate)))
            in_features = int((in_features + self.num_layers * self.growth_rate)*self.compression_rate)

        len3 = math.floor((input_len-9)/5)+1
        for a in range(self.dense_block_num-1):
            len3 = math.floor((len3-9)/5)+1
        self.linears2 = nn.Sequential(
            nn.Linear(len3*53,64),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(64,1)
        )
            
    def forward(self, x, y_class=None):
        x = self.embedding(x)
        max_x = self.maxpool(x)
        avg_x = self.avgpool(x)
        
        x_col = torch.cat((max_x, avg_x), 2)
        x_col = x_col.permute(0, 2, 1)
        x_col = self.Conv1d_2(x_col)
        x_col = torch.squeeze(x_col, 1)
        x_att = self.sigmoid(x_col).unsqueeze(2)
        x_att = torch.tile(x_att,(1,1,32))
        x = torch.mul(x, x_att)
        
        x = x.permute(0, 2, 1)
        x = self.Conv1d_1(x)
        x = nn.Dropout(0.2)(x)
        x = x.permute(0, 2, 1)
        #x,_ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.ConvCaps2D(x)
        x1 = self.routing_caps(x)
        classes = torch.sqrt(torch.sum(x1**2, dim=-1))

        mask = self.mask(x1, y_class)
        x1 = x1.view(x1.size(0),  -1)
        x_gen = self.generator(x1)

        #x1, x = self.digitCaps(x)
        x = self.linears1(x1)

        x_gen = x_gen.permute(0, 2, 1)
        x_gen1 = self.dense_layer[0](x_gen)
        for layer in self.dense_layer[1:]:
            x_gen1=layer(x_gen1)
        x_gen1 = x_gen1.view(x_gen1.size(0), -1)
        x2 = self.linears2(x_gen1)
        x_gen = x_gen.permute(0, 2, 1)

        return x, x_att, x_gen, x2
        #return x, x_gen, torch.sigmoid(classes)

