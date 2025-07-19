import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
import models.resnet18_swat
from models.resnet20_cifar import *
import models
import logging
import math

# +++ START OF ADDED 1D-RESNET BLOCKS +++
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet1D, self).__init__()
        self.in_planes = 64 # Corresponds to RESNET_BLOCK_CHANNELS[0]
        
        # Input is expected to be (batch, 1, features)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        # Flatten the features
        out = out.view(out.size(0), -1)
        return out
# +++ END OF ADDED 1D-RESNET BLOCKS +++


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        self.initial_conv = None # For CVXPY weights
        self.initial_bn = None  # For CVXPY weights

        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        elif self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000', 'mini_imagenet_withpath']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        elif self.args.dataset in ['cub200','manyshotcub']:
            self.encoder = resnet18(True, args)  # pretrained=True
            self.num_features = 512
        # +++ START OF MODIFICATION FOR SWAT DATASET +++
        elif self.args.dataset in ['swat']:
            # This linear layer will be initialized by the CVXPY weights
            if self.args.is_pretrain:
                # 这个卷积层将作为可解释的预处理层，在encoder之前
                self.initial_conv = nn.Conv1d(
                    in_channels=args.num_features_input,
                    out_channels=args.num_features_input,
                    kernel_size=1,
                    bias=False
                )
                self.initial_bn = nn.BatchNorm1d(args.num_features_input)
            
            # The encoder is the 1D ResNet which follows the initial linear layer
            # num_blocks matches the standalone script's [2, 2, 2, 2]
            self.encoder = ResNet1D(BasicBlock1D, num_blocks=[2, 2, 2, 2])
            
            # The output feature dimension from the 1D ResNet is 512
            self.num_features = 512
        # +++ END OF MODIFICATION FOR SWAT DATASET +++
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.dropout_fn = nn.Dropout(0.3)

    def forward_metric(self, x):
        x_feature = self.encode(x)
        if 'cos' in self.mode:
            if self.dropout_fn is not None:
                x_feature = self.dropout_fn(x_feature)
            x = F.linear(F.normalize(x_feature, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x
        elif 'dot' in self.mode:
            x = self.fc(x_feature)
        return x_feature, x

    def encode(self, x):
        # +++ START OF MODIFICATION FOR SWAT DATASET +++
        if self.args.dataset == 'swat':
            if self.args.is_pretrain and self.initial_conv is not None:
                # Pass through the CVXPY-initialized linear layer first
                out = x.unsqueeze(-1)
                out = self.initial_conv(out)
                out = self.initial_bn(out)
                out = out.squeeze(-1)  # Remove the last dimension
            else:
                out = x

            out = out.unsqueeze(1)  # Add a channel dimension for 1D Conv
            # Pass through the 1D-ResNet encoder
            return self.encoder(out)
        # +++ END OF MODIFICATION FOR SWAT DATASET +++
        else: # Original logic for 2D image data
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
            return x

    def set_initial_weights(self, W_init_tensor):
        # CVXPY返回的W_init_tensor是(51, 51)
        # 我们需要将其变形为 (out_channels, in_channels, kernel_size) -> (51, 51, 1)
        if W_init_tensor.dim() == 2:
            W_init_tensor_conv = W_init_tensor.unsqueeze(-1)
        else:
            W_init_tensor_conv = W_init_tensor
        
        if W_init_tensor_conv.shape == self.initial_conv.weight.shape:
            with torch.no_grad():
                self.initial_conv.weight.copy_(W_init_tensor_conv)
            print("已成功从CVXPY设置 MYNET 的 initial_conv 层权重。")
        else:
            print(f"形状不匹配: W_init_tensor {W_init_tensor_conv.shape}, initial_conv.weight {self.initial_conv.weight.shape}。权重未设置。")

    def forward(self, input):
        if self.mode != 'encoder':
            x_feature, logits = self.forward_metric(input)
            return x_feature, logits
        elif self.mode == 'encoder':
            return self.encode(input)
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'finetune' in self.args.new_mode:
            self.update_fc_ft(new_fc,data,label,session)


    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)
        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)