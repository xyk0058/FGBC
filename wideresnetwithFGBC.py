from ast import Not
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np

class Relation(nn.Module):
    def __init__(self, in_features):
        super(Relation, self).__init__()
        
        self.gamma_1 = nn.Linear(in_features, in_features, bias=False)
        self.gamma_2 = nn.Linear(in_features, in_features, bias=False)

        self.beta_1 = nn.Linear(in_features, in_features, bias=False)
        self.beta_2 = nn.Linear(in_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, ft, neighbor):
        gamma = self.gamma_1(ft) + self.gamma_2(neighbor)
        gamma = self.lrelu(gamma) + 1.0

        beta = self.beta_1(ft) + self.beta_2(neighbor)
        beta = self.lrelu(beta)

        self.r_v = gamma * self.r + beta

        #transE
        self.m = ft + self.r_v - neighbor
        '''
        #transH
        norm = F.normalize(self.r_v) 
        h_ft = ft - norm * torch.sum((norm * ft), dim=1, keepdim=True)
        h_neighbor = neighbor - norm * torch.sum((norm * neighbor), dim=1, keepdim=True)
        self.m = h_ft - h_neighbor
        '''
        return self.m #F.normalize(self.m)

class GPR_prop(torch.nn.Module):
    '''
    propagation class for GPR_GNN
    '''
    def __init__(self, K, alpha=0.1):
        super(GPR_prop, self).__init__()
        self.K = K
        self.alpha = alpha
        bound = np.sqrt(3/(K+1))
        TEMP = np.random.uniform(-bound, bound, K+1)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        self.temp = nn.Parameter(torch.FloatTensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K
    
    def forward(self, x, A_hat):
        num_nodes, d_model = x.size()
        output = self.temp[0] * x
        for i in range(self.K):
            x = A_hat @ x
            output = output + self.temp[i+1] * x
        return output


class GPRGNN(torch.nn.Module):
    def __init__(self, K, input_dim, d_model, output_dim):
        super(GPRGNN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, d_model, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(),
            nn.Linear(d_model, output_dim)
            # nn.Linear(input_dim, output_dim)
        )
        # self.linear_out = nn.Sequential(
        #     nn.Linear(d_model, output_dim)
        # )
        self.fc = nn.Linear(input_dim, input_dim, bias=True)
        self.prop = GPR_prop(K)
        self.d_model = d_model
        self.prop.reset_parameters()
        self.num_classes = output_dim
        self.r = Relation(input_dim)
        self.output_dim = output_dim

    # def forward(self, x, cal_rn_weight=False):
    #     x = self.linear(x)
    #     return x, x, None, None#output_r

    def getLinear(self, x):
        return self.linear(x)
    
    def forward(self, x, cal_rn_weight=False):
        # x = x.detach()
        A_hat = torch.softmax(self.fc(x) @ x.T, dim=-1) + torch.eye(x.shape[0], device=x.device)
        # A_hat = F.normalize(self.fc(x) @ x.T, p=1, dim=1) + torch.eye(x.shape[0], device=x.device)
        # neighbor = torch.mm(A_hat, x)
        # output_r = self.r(x, neighbor)

        x = self.linear(x)
        out = self.prop(x, A_hat)

        labels = torch.argmax(F.softmax(out, dim=1), dim=1).detach()
        # out = out + torch.where(labels < (self.output_dim / 2), 0, 1).reshape(labels.shape[0],1) * self.linear(output_r)
        # out2 = out + self.linear(output_r)
        # num_neighbor = torch.sum(A_hat, dim=1, keepdim=True)
        # out2 = out2 / (num_neighbor+1)

        # select_out = torch.where(labels < 5, 1, 0).reshape(labels.shape[0],1).detach()
        # out = select_out * out + (1-select_out) * out2
        

        rn_weight = None

        if cal_rn_weight:
            # labels = torch.argmax(targets_x2, dim=1)

            ppr_dense = A_hat
            # gpr_dense = torch.zeros((self.num_classes, labels.shape[0])).float().cuda()

            # for iter_c in range(self.num_classes):
            #     iter_where = torch.where(labels==iter_c)[0]
            #     iter_mean  = torch.mean(ppr_dense[iter_where],dim=0)
            #     gpr_dense[iter_c] = iter_mean

            # gpr_dense = gpr_dense.transpose(0,1)
            # gpr_sum = torch.sum(gpr_dense,dim=1)
            # gpr_idx = F.one_hot(torch.tensor(labels).long(), self.num_classes).float().cuda()

            # gpr_rn = gpr_sum.unsqueeze(1) - gpr_dense
            # rn_dense = torch.mm(ppr_dense,gpr_rn)
            # rn_value = torch.sum(rn_dense * gpr_idx, dim=1)

            # totoro_list = rn_value.tolist()
            # nnode = len(totoro_list)
            # train_size = len(totoro_list)

            # id2totoro = {i:totoro_list[i] for i in range(len(totoro_list))}
            # sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
            # id2rank = {sorted_totoro[i][0]:i for i in range(nnode)}
            # totoro_rank = [id2rank[i] for i in range(nnode)]

            # base_w = 0.75                       # the base  value for ReNode re-weighting; value set to [0.25,0.5,0.75,1]
            # scale_w = 0.5                       # the scale value for ReNode re-weighting; value set to [1.5 ,1  ,0.5 ,0]
            # rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
            # rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor).cuda().detach()

        return out, x, rn_weight, None#output_r



# class GPRGNN(torch.nn.Module):
#     def __init__(self, K, input_dim, d_model, output_dim):
#         super(GPRGNN, self).__init__()
#         self.layer1 = GPR(K=2, input_dim=input_dim, d_model=d_model, output_dim=d_model)
#         self.layer2 = GPR(K=1, input_dim=d_model, d_model=d_model, output_dim=output_dim)
    
#     def forward(self, x, cal_rn_weight=False):
#         out, _, _ = self.layer1(x, cal_rn_weight=False)
#         out, _, rn_weight = self.layer2(out, cal_rn_weight=True)
#         return out, x, rn_weight

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0, K=3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.rot = nn.Linear(nChannels[3], 4)
        self.fc2=nn.Linear(nChannels[3], num_classes)
        #self.hyper=Variable(torch.rand(10).type(torch.FloatTensor),requires_grad=True).cuda(0)

        self.graph_model0 = GPRGNN(K=1, input_dim=nChannels[3], d_model=256, output_dim=num_classes)
        self.graph_model = GPRGNN(K=1, input_dim=nChannels[3], d_model=256, output_dim=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out=self.bn1(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out
    
    def getLinear(self, x):
        return self.graph_model.getLinear(x)

    def warmup(self, epoch, max_val=1, mult=-5, max_epochs=500):
        if epoch == 0:
            return 0.
        w = max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)
        w = float(w)
        if epoch > max_epochs:
            return max_val
        return w

    def classify(self,out, cal_rn_weight=False):
        # return self.fc(out)
        after_gnn, before_gnn, rn_weight, output_r = self.graph_model0(out, cal_rn_weight)
        if cal_rn_weight:
            return after_gnn, rn_weight
        return after_gnn
    
    def classify2(self, out, cal_rn_weight=False):
        after_gnn, before_gnn, rn_weight, output_r = self.graph_model(out, cal_rn_weight)
        
        if self.training:
            return after_gnn, before_gnn, rn_weight
        else:
            return after_gnn


    # def classify(self,out, cal_rn_weight=False):
    #     return self.fc(out)
    # def classify2(self, out, cal_rn_weight=False):
    #     return self.fc2(out)


    def rotclassify(self,out):
        return self.rot(out)

    

    #def hyperprod(self,out):
     #   return F.normalize(out*self.hyper,p=1,dim=1)
