'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
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
            nn.Linear(input_dim, input_dim * 2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(),
            nn.Linear(input_dim * 2, output_dim)
            # nn.Linear(input_dim, output_dim)
        )
        # self.linear_out = nn.Sequential(
        #     nn.Linear(d_model, output_dim)
        # )
        # self.fc = nn.Linear(input_dim, input_dim, bias=True)
        self.prop = GPR_prop(K)
        self.d_model = d_model
        self.prop.reset_parameters()
        self.num_classes = output_dim
        # self.r = Relation(input_dim)
        self.output_dim = output_dim

    
    def forward(self, x, cal_rn_weight=False):
        # x = x.detach()
        A_hat = torch.softmax(x @ x.T, dim=-1) + torch.eye(x.shape[0]).cuda()

        x = self.linear(x)
        out = self.prop(x, A_hat)

        labels = torch.argmax(F.softmax(out, dim=1), dim=1).detach()

        rn_weight = None

        if cal_rn_weight:
            # labels = torch.argmax(targets_x2, dim=1)
            ppr_dense = A_hat
            gpr_dense = torch.zeros((self.num_classes, labels.shape[0])).float().cuda()

            for iter_c in range(self.num_classes):
                iter_where = torch.where(labels==iter_c)[0]
                iter_mean  = torch.mean(ppr_dense[iter_where],dim=0)
                gpr_dense[iter_c] = iter_mean

            gpr_dense = gpr_dense.transpose(0,1)
            gpr_sum = torch.sum(gpr_dense,dim=1)
            gpr_idx = F.one_hot(torch.tensor(labels).long(), self.num_classes).float().cuda()

            gpr_rn = gpr_sum.unsqueeze(1) - gpr_dense
            rn_dense = torch.mm(ppr_dense,gpr_rn)
            rn_value = torch.sum(rn_dense * gpr_idx, dim=1)

            totoro_list = rn_value.tolist()
            nnode = len(totoro_list)
            train_size = len(totoro_list)

            id2totoro = {i:totoro_list[i] for i in range(len(totoro_list))}
            sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
            id2rank = {sorted_totoro[i][0]:i for i in range(nnode)}
            totoro_rank = [id2rank[i] for i in range(nnode)]

            base_w = 0.75                       # the base  value for ReNode re-weighting; value set to [0.25,0.5,0.75,1]
            scale_w = 0.5                       # the scale value for ReNode re-weighting; value set to [1.5 ,1  ,0.5 ,0]
            rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
            rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor).cuda().detach()

        return out, x, rn_weight



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rotation=True, rotnet=None, classifier_bias=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.output1 = nn.Linear(512*block.expansion, num_classes, bias=classifier_bias)
        self.output2 = nn.Linear(512*block.expansion, num_classes, bias=classifier_bias)

        # self.graph_model0 = GPRGNN(K=1, input_dim=512*block.expansion, d_model=256, output_dim=num_classes)
        # self.graph_model = GPRGNN(K=1, input_dim=512*block.expansion, d_model=256, output_dim=num_classes)


        self.rotation = rotation
        if self.rotation:
            if rotnet is not None:
                self.rot = rotnet
            else:
                self.rot = nn.Linear(512*block.expansion, 4)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        f = F.adaptive_avg_pool2d(out, 1)

        out = f.squeeze()
        return out
        # c = self.output(f.squeeze())

        # if self.rotation:
        #     r = self.rot(f.squeeze())
        # else:
        #     r = 0

        # if return_feature:
        #     return [c, r, f]
        # else:
        #     return c, r
    
    def classify(self,out, cal_rn_weight=False):
        # after_gnn, before_gnn, rn_weight = self.graph_model0(out, cal_rn_weight)
        # if cal_rn_weight:
        #     return after_gnn, rn_weight
        # return after_gnn
        if cal_rn_weight:
            return self.output1(out), None
        return self.output1(out)

    def rotclassify(self,out):
        return self.rot(out)

    def classify2(self, out, cal_rn_weight=False):
        # after_gnn, before_gnn, rn_weight = self.graph_model(out, cal_rn_weight)
        
        # if self.training:
        #     return after_gnn, before_gnn, rn_weight
        # else:
        #     return after_gnn
        if self.training:
            return self.output2(out), None
        else:
            return self.output2(out)

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
