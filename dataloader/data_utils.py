import numpy as np
import torch
from dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'swat':
        import dataloader.swat.swat as Dataset
        args.base_class = 21
        args.num_classes= 36
        args.way = 5
        args.shot = 5
        args.sessions = 4
    if args.dataset == 'wadi':
        import dataloader.wadi.wadi as Dataset
        args.base_class = 8
        args.num_classes= 14
        args.way = 1
        args.shot = 5
        args.sessions = 7
    if args.dataset == 'hai':
        import dataloader.hai.hai as Dataset
        args.base_class = 39
        args.num_classes= 59
        args.way = 2
        args.shot = 5
        args.sessions = 11
    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'swat':
        trainset = args.Dataset.Swat(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.Swat(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'wadi':
        trainset = args.Dataset.Wadi(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.Wadi(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'hai':
        trainset = args.Dataset.HAI(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.HAI(root=args.dataroot, train=False, index=class_index)


    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'swat':
        trainset = args.Dataset.Swat(root=args.dataroot, train=True,
                                       index=np.arange(args.base_class + (session-1) * args.way, args.base_class + session * args.way))
    if args.dataset == 'wadi':
        trainset = args.Dataset.Wadi(root=args.dataroot, train=True,
                                       index=np.arange(args.base_class + (session-1) * args.way, args.base_class + session * args.way))
    if args.dataset == 'hai':
        trainset = args.Dataset.HAI(root=args.dataroot, train=True,
                                       index=np.arange(args.base_class + (session-1) * args.way, args.base_class + session * args.way))
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'swat':
        testset = args.Dataset.Swat(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'wadi':
        testset = args.Dataset.Wadi(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'hai':
        testset = args.Dataset.HAI(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=True,
                                             num_workers=args.num_workers)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list