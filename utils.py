import json
import os
import pprint as pprint
import random
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from sklearn.metrics import confusion_matrix
import logging
from logging.config import dictConfig
from dataloader.data_utils import *
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from models import *



_utils_pp = pprint.PrettyPrinter()

def set_logging(level, work_dir):
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": f"%(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": f"{level}",
                "class": "logging.StreamHandler",
                'formatter': 'simple',
            },
            'file': {
                'level': f"{level}",
                'formatter': 'simple',
                'class': 'logging.FileHandler',
                'filename': f'{work_dir if work_dir is not None else "."}/train.log',
                'mode': 'a',
            },
        },
        "loggers": {
            "": {
                "level": f"{level}",
                "handlers": ["console", "file"] if work_dir is not None else ["console"],
            },
        },
    }
    dictConfig(LOGGING)
    logging.info(f"Log level set to: {level}")

def pprint(x):
    _utils_pp.pprint(x)
class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)
class Logger(object):
    def __init__(self, args, log_dir, **kwargs):
        self.logger_path = os.path.join(log_dir, 'scalars.json')
        # self.tb_logger = SummaryWriter(
        #                     logdir=osp.join(log_dir, 'tflogger'),
        #                     **kwargs,
        #                     )
        self.log_config(vars(args))

        self.scalars = defaultdict(OrderedDict) 

    # def add_scalar(self, key, value, counter):
    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        # self.tb_logger.add_scalar(key, value, counter)

    def log_config(self, variant_data):
        config_filepath = os.path.join(os.path.dirname(self.logger_path), 'configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def dump(self):
        with open(self.logger_path, 'w') as fd:
            json.dump(self.scalars, fd, indent=2)

def set_seed(seed):
    if seed == 0:
        logging.info(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        logging.info('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    logging.info('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        logging.info('create folder:', path)
        os.makedirs(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    def __repr__(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1) 
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)

def count_acc_taskIL(logits, label,args):
    basenum=args.base_class
    incrementnum=(args.num_classes-args.base_class)/args.way
    for i in range(len(label)):
        currentlabel=label[i]
        if currentlabel<basenum:
            logits[i,basenum:]=-1e9
        else:
            space=int((currentlabel-basenum)/args.way)
            low=basenum+space*args.way
            high=low+args.way
            logits[i,:low]=-1e9
            logits[i,high:]=-1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def confmatrix(logits,label):
    
    font={'family':'FreeSerif','size':18}
    matplotlib.rc('font',**font)
    matplotlib.rcParams.update({'font.family':'FreeSerif','font.size':18})
    plt.rcParams["font.family"]="FreeSerif"

    pred = torch.argmax(logits, dim=1)
    cm=confusion_matrix(label, pred, normalize='true')

    return cm

def save_list_to_txt(name, input_list):
    f = open(name, mode='a+')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()
    
def postprocess_results(result_list, trlog):
        result_list.append('Base Session Best Epoch {}\n'.format(trlog['max_acc_epoch']))
        result_list.append(trlog['max_acc'])
        result_list.append("Seen acc:")
        result_list.append(trlog['seen_acc'])
        result_list.append('Unseen acc:')
        result_list.append(trlog['unseen_acc'])
        hmeans = harm_mean(trlog['seen_acc'], trlog['unseen_acc'])
        result_list.append('Harmonic mean:')
        result_list.append(hmeans)

        logging.info(f"max_acc: {trlog['max_acc']}")        
        logging.info(f"Unseen acc: {trlog['unseen_acc']}")
        logging.info(f"Seen acc: {trlog['seen_acc']}")
        logging.info(f"Harmonic mean: {hmeans}")
        return result_list, hmeans

def save_result(args, trlog, hmeans, **kwargs):
    params_info = args.save_path.split('/')[-1]
    main_path = f"results/main/{args.project}"
    os.makedirs(main_path, exist_ok=True)
    details_path = f"results/details/{args.project}"
    os.makedirs(details_path, exist_ok=True)
    with open(os.path.join(main_path, f"{args.dataset}_results.csv"), "a+") as f:
        f.write(f"{params_info}-{trlog['max_acc'][0]},{trlog['max_acc'][-1]},{trlog['unseen_acc'][0]},{trlog['unseen_acc'][-1]},{hmeans[0]},{hmeans[-1]},{args.time_str} \n")
    with open(os.path.join(details_path, f"{args.dataset}_results.csv"), "a+") as f:
        f.write(f">>> {params_info}-Avg_acc:{trlog['max_acc']} \n Seen_acc:{trlog['seen_acc']} \n Unseen_acc:{trlog['unseen_acc']} \n HMean_acc:{hmeans} \n")

def harm_mean(seen, unseen):
    # compute from session1
    assert len(seen) == len(unseen)
    harm_means = []
    for _seen, _unseen in zip(seen, unseen):
        _hmean = (2 * _seen * _unseen) / (_seen + _unseen + 1e-12)
        _hmean = float('%.3f' % (_hmean))
        harm_means.append(_hmean)
    return harm_means

def get_optimizer(args, model, **kwargs):
        # prepare optimizer
        if args.project in ['teen', 'warp']:
            if args.optim == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), args.lr_base, 
                                            momentum=args.momentum, nesterov=True,
                                            weight_decay=args.decay)
            elif args.optim == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), 
                                             lr=args.lr_base, weight_decay=args.decay)
        
        
        # prepare scheduler
        if args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, 
                                                        gamma=args.gamma)
        elif args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                             gamma=args.gamma)
        elif args.schedule == 'Cosine':
            assert args.tmax >= 0 , "args.tmax should be greater than 0"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)
        return optimizer, scheduler


def get_features(loader, transform, model):
    model = model.eval()

    loader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(loader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    np.save('embedding_list.npy', embedding_list.numpy())
    np.save('label_list.npy', label_list.numpy())
    return embedding_list, label_list

def save_s_tne(features, labels):
    # 创建 t-SNE 实例并拟合数据
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    test_features_tsne = tsne.fit_transform(features)

    np.save('features_tsne.npy', test_features_tsne)
    np.save('label_list.npy', labels)



def compute_orthonormal(args, net, trainset):
    training = net.training
    indices = torch.randperm(len(trainset))[:20 * args.batch_size_base] # 随机选择一部分数据计算正交基


    trainset = torch.utils.data.Subset(trainset, indices)
    dl = DataLoader(trainset, shuffle=False, batch_size=args.batch_size_base)
    net.eval()
    net.forward_covariance = None
    net.batch_count = 0

    cdmodule_list = [module for module in net.modules() if isinstance(module, WaRPModule)] # 获取所有的WaRP模块
    for m in cdmodule_list:
        m.flag = False
    epoch_iter = tqdm(dl)

    with torch.no_grad():
        for x, y in epoch_iter:
            x, y = x.cuda(), y.cuda()
            encoded_x = net.module.encode(x)[:, :args.base_class] # 进行前向传播，以便后续计算warp模块的激活协方差
            for m in cdmodule_list:
                m.post_backward() # 计算每个warp模块的激活协方差

        module_it = tqdm(cdmodule_list)
        for m in module_it:
            m.flag = True
            forward_cov = getattr(m, 'forward_covariance')

            V, S, UT_forward = torch.linalg.svd(forward_cov, full_matrices=True) # 奇异值分解

            weight = getattr(m, 'weight')
            bias = getattr(m, 'bias')
            if weight.ndim != 2:
                weight = weight.reshape(weight.shape[0], -1)

            UT_backward = torch.eye(weight.shape[0]) # 单位矩阵
            UT_backward = same_device(ensure_tensor(UT_backward), weight) # 保证UT_backward和weight在同一设备上


            UT_forward = same_device(UT_forward, weight)
            basis_coefficients = UT_backward @ weight @ UT_forward.t() # 新基底的参数
            m.UT_forward = UT_forward
            m.UT_backward = UT_backward
            m.basis_coefficients.data = basis_coefficients.data
            m.basis_coeff.data = basis_coefficients.data

        for m in module_it:

            coefficients = getattr(m, 'basis_coefficients')
            UT_forward = getattr(m, 'UT_forward')
            UT_backward = getattr(m, 'UT_backward')
            if m.weight.ndim != 2:
                basis_coefficients = coefficients.reshape(m.weight.shape[0],
                                                    UT_forward.shape[0],
                                                    1, 1)
                m.UT_forward_conv = UT_forward.reshape(UT_forward.shape[0],
                                                               m.weight.shape[1],
                                                               m.weight.shape[2],
                                                               m.weight.shape[3])
                m.UT_backward_conv = UT_backward.t().reshape(m.weight.shape[0],
                                                                     m.weight.shape[0],
                                                                     1,
                                                                     1)
                m.basis_coeff.data = basis_coefficients.data

    net.training = training
    print(f'Compute orthonormal is completed!')


def identify_importance(args, model, trainset, batchsize=60, keep_ratio=0.1, session=0, way=10, new_labels=None):
    importances = OrderedDict()
    temp = OrderedDict()
    dl = DataLoader(trainset, shuffle=False, batch_size=batchsize)
    model.eval().cuda()

    for module in model.modules():
        if isinstance(module, WaRPModule):

            module.coeff_mask_prev = module.coeff_mask.data # 记录上一个任务的掩码
            module.coeff_mask.data = torch.zeros(module.coeff_mask.shape).cuda().data

    training = model.training

    epoch_iter = tqdm(dl)
    for i, batch in enumerate(epoch_iter):
        if session == 0:
            x, y = [_.cuda() for _ in batch]
        else:
            x, y = batch.cuda(), new_labels.cuda()[i * batchsize:(i+1) * batchsize]
        yhat = model(x)[:, :args.base_class + session * way]
        loss = nn.CrossEntropyLoss()(yhat, y)
        model.zero_grad()
        loss.backward()

        for module in model.modules():
            if isinstance(module, WaRPModule):
                temp[module] = module.basis_coeff.grad.abs().detach().cpu().numpy().copy() # 记录新基底系数梯度的绝对值

        for module in model.modules():
            if isinstance(module, WaRPModule):
                if module not in importances:
                    importances[module] = temp[module] # 梯度的绝对值求和
                else:
                    importances[module] += temp[module]


    flat_importances = flatten_importances_module(importances)
    threshold = fraction_threshold(flat_importances, keep_ratio)
    masks = importance_masks_module(importances, threshold)


    # 合并之前任务的掩码，也就是保留所有关于旧知识的参数
    for module in model.modules():
        if isinstance(module, WaRPModule):
            coeff_mask = masks[module]
            coeff_mask = same_device(ensure_tensor(coeff_mask), module.basis_coefficients)
            # module.coeff_mask.data = 1 - (1 - coeff_mask.data) * (1 - module.coeff_mask_prev.data)
            module.coeff_mask.data = torch.clamp(coeff_mask.data + module.coeff_mask_prev.data, min=0, max=1)
            # softmax_coeff_mask = importances[module]
            # softmax_coeff_mask = same_device(ensure_tensor(softmax_coeff_mask), module.basis_coefficients)
            # module.coeff_mask.data = torch.clamp(coeff_mask + module.coeff_mask_prev.data + importances_softmax(softmax_coeff_mask.data), min=0, max=1)
            # coeff_mask = importances[module]
            # coeff_mask = same_device(ensure_tensor(coeff_mask), module.basis_coefficients)
            # module.coeff_mask.data = importances_softmax(coeff_mask.data)


    # 保存新合成的掩码
    # -------------------------- get accumulative mask ratio ---------------------------------
    for module in model.modules():
        if isinstance(module, WaRPModule):
            masks[module] = module.coeff_mask.data.detach().cpu().numpy().copy()
    print(flatten_importances_module(masks).mean())
    # ----------------------------------------------------------------------------------------


    model.zero_grad()
    model.training = training

    # 对于非WaRP模块，冻结参数
    for module in model.modules():
        if hasattr(module, 'weight') and not isinstance(module, WaRPModule):
            for param in module.parameters():
                param.requires_grad = False

    print(f'The identify importance of {session} is computed!')
    return model




# 将所有参数展平，输出一个一维数组
def flatten_importances_module(importances):
    return np.concatenate([
        params.flatten()
        for _, params in importances.items()
    ])

# 返回每个模块的重要性
def map_importances_module_dict(fn, importances):
    return {module: fn(params)
            for module, params in importances.items()}

# 返回一个字典，记录每个模块相应的参数掩码
def importance_masks_module(importances, threshold):
    return map_importances_module_dict(lambda imp: threshold_mask(imp, threshold), importances)


# 通过参数保留比率计算阈值
def fraction_threshold(tensor, fraction):
    """Compute threshold quantile for a given scoring function

    Given a tensor and a fraction of parameters to keep,
    computes the quantile so that only the specified fraction
    are larger than said threshold after applying a given scoring
    function. By default, magnitude pruning is applied so absolute value
    is used.

    Arguments:
        tensor {numpy.ndarray} -- Tensor to compute threshold for
        fraction {float} -- Fraction of parameters to keep

    Returns:
        float -- Threshold
    """
    assert isinstance(tensor, np.ndarray)
    threshold = np.quantile(tensor, 1-fraction)
    return threshold

# 根据给定阈值生成二值掩码
def threshold_mask(tensor, threshold):
    """Given a fraction or threshold, compute binary mask

    Arguments:
        tensor {numpy.ndarray} -- Array to compute the mask for

    Keyword Arguments:
        threshold {float} -- Absolute threshold for dropping params

    Returns:
        np.ndarray -- Binary mask
    """
    assert isinstance(tensor, np.ndarray)
    idx = np.logical_and(tensor < threshold, tensor > -threshold)
    mask = np.ones_like(tensor)
    mask[idx] = tensor[idx]
    return mask



# 将权重空间变换为原来的
def restore_weight(net):
    cdmodule_list = [module for module in net.modules() if isinstance(module, WaRPModule)]
    for module in cdmodule_list:
        weight = getattr(module, 'weight')
        UT_forward = getattr(module, 'UT_forward')
        UT_backward = getattr(module, 'UT_backward')
        coeff_mask = getattr(module, 'coeff_mask').reshape(weight.shape[0], -1)

        coeff_mask = same_device(ensure_tensor(coeff_mask), module.basis_coeff.data)
        weight_res = UT_backward.t() @ module.basis_coeff.data.reshape(coeff_mask.shape) @ UT_forward
        weight_res = weight_res.reshape(weight.shape)
        module.weight.data = weight_res.data
    return net


def compute_accum_ratio(model, session):
    masks = OrderedDict()
    for module in model.modules():
        if isinstance(module, WaRPModule):
            masks[module] = module.coeff_mask.data.detach().cpu().numpy().copy()
    logs = dict(num_session=session, keep_ratio=flatten_importances_module(masks).mean().item())
    return logs

def importances_softmax(importances):
    importances_flatten = importances.view(-1)
    importances_flatten_softmax = torch.exp(importances_flatten) / torch.exp(torch.max(importances_flatten))
    return importances_flatten_softmax.view(importances.shape)