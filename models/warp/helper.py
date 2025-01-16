# import new Network name here and add in model_class args
import time

from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)


    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)

        logits_ = logits[:, :args.base_class]
        loss = F.cross_entropy(logits_, train_label)
        acc = count_acc(logits_, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item())
        ta.add(acc)
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), ta.item()))

        optimizer.zero_grad()
        # loss.backward()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()


    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    logging.info(get_accuracy_per_class(model, testloader, test_class))
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class] # 选择前 test_class 个类别的预测结果
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            base_idxs = test_label < args.base_class
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs]) # 只预测基类的情况下基类预测正确的数量
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs]) # 给出新类的情况下基类预测正确的数量
                va_base.add(acc_base) # 不断计算平均准确率
                va_base_given_new.add(acc_base_given_new)


            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class) # 只预测新类的情况下新类预测正确的数量
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs]) # 给出基类的情况下基类预测正确的数量
                va_new.add(acc_new)
                va_new_given_base.add(acc_new_given_base)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
    logging.info('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    logging.info('base only accuracy: {:.4f}, new only accuracy: {:.4f}'.format(va_base, va_new))
    logging.info('base acc given new : {:.4f}'.format(va_base_given_new))
    logging.info('new acc given base : {:.4f}'.format(va_new_given_base))

    logs = dict(num_session=session + 1, acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new,
                new_acc_given_base=va_new_given_base)

    return vl, va, logs

import torch

def get_accuracy_per_class(model, testloader, num_classes):
    model.eval()
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in testloader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(num_classes):
                class_mask = (labels == i)
                correct_per_class[i] += (preds[class_mask] == labels[class_mask]).sum().item()
                total_per_class[i] += class_mask.sum().item()

    acc_per_class = [correct / total if total != 0 else 0 for correct, total in zip(correct_per_class, total_per_class)]
    return acc_per_class


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