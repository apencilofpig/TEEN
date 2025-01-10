import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
import models.resnet18_swat
from models.resnet20_cifar import *
import models


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000', 'mini_imagenet_withpath']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200','manyshotcub']:
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        if self.args.dataset in ['swat', 'wadi']:
            # self.encoder = nn.Sequential(
            #     nn.Conv2d(1, 64, 3),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.3),
            #     # nn.Conv2d(32, 64, 3),
            #     # nn.BatchNorm2d(64),
            #     # nn.ReLU(),
            #     # nn.MaxPool2d(2, stride=2),
            #     # nn.Dropout(0.3),
            #     nn.Conv2d(64, 512, 3),
            #     nn.BatchNorm2d(512),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.3)
            # )
            self.encoder = nn.Sequential(
                nn.Embedding(256, 32),
                models.resnet18_swat.resnet18(False, args)
            )
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.centers = nn.Parameter(torch.randn(args.multi_proto_num, self.num_features)).to('cuda')
        self.dropout_fn = nn.Dropout(0.3)

    def forward_metric(self, x):
        x_feature = self.encode(x)
        if 'cos' in self.mode:
            if self.dropout_fn is None:
                x = F.linear(F.normalize(x_feature, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            else:
                x = F.linear(self.dropout_fn(F.normalize(x_feature, p=2, dim=-1)), F.normalize(self.fc.weight, p=2, dim=-1))

            # x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x_feature)
            # x = self.args.temperature * x
        return x_feature, x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            x_feature, input = self.forward_metric(input)
            return x_feature, input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        global data_imgs
        for batch in dataloader:
            data_imgs, label = [_.cuda() for _ in batch]
            data=self.encode(data_imgs).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        print('further finetune?')
        self.update_fc_ft(new_fc,data_imgs,label,session, class_list)

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
        
    # def update_fc_ft(self, new_fc, data_imgs,label,session, class_list=None):
    #     self.eval()
    #     optimizer_embedding = torch.optim.SGD(self.encoder.parameters(), lr=self.args.lr_new, momentum=0.9)

    #     with torch.enable_grad():
    #         for epoch in range(self.args.epochs_new):


    #             fc = self.fc.weight[:self.args.base_class + self.args.way * session, :].detach()
    #             data = self.encode(data_imgs)
    #             logits = self.get_logits(data, fc)
    #             # acc = count_acc(logits, label)

    #             loss = F.cross_entropy(logits, label)
    #             optimizer_embedding.zero_grad()
    #             loss.backward()

    #             optimizer_embedding.step()

    def update_fc_ft(self,new_fc,data_imgs,label,session, class_list=None):
        new_fc=new_fc.clone().detach()
        print(new_fc)
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                data = self.encode(data_imgs)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(new_fc)
        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)



    def soft_calibration(self, args, session):
        base_protos = self.fc.weight.data[:args.base_class].detach().cpu().data
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        
        cur_protos = self.fc.weight.data[args.base_class + (session-1) * args.way : args.base_class + session * args.way].detach().cpu().data
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        
        weights = torch.mm(cur_protos, base_protos.T) * args.softmax_t
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)

        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        
        updated_protos = (1-args.shift_weight) * cur_protos + args.shift_weight * delta_protos

        self.fc.weight.data[args.base_class + (session-1) * args.way : args.base_class + session * args.way] = updated_protos