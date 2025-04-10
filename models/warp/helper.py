# import new Network name here and add in model_class args
import time

from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



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
    vacc = Averager()  # 准确率平均
    vprecision = Averager()  # 精确率平均
    vrecall = Averager()  # 召回率平均
    vf1 = Averager()  # F1平均

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

            labels = test_label.cpu().numpy()
            _, preds = torch.max(logits, 1)
            preds = preds.cpu().numpy()

            vacc.add(accuracy_score(labels, preds))
            vprecision.add(precision_score(labels, preds, average='macro', zero_division=0))
            vrecall.add(recall_score(labels, preds, average='macro', zero_division=0))
            vf1.add(f1_score(labels, preds, average='macro', zero_division=0))

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
        vacc = vacc.item()
        vprecision = vprecision.item()
        vrecall = vrecall.item()
        vf1 = vf1.item()
    logging.info('epo {}, test, loss={:.4f} acc={:.4f}, accuracy={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}'.format(epoch, vl, va, vacc, vprecision, vrecall, vf1))
    logging.info('base only accuracy: {:.4f}, new only accuracy: {:.4f}'.format(va_base, va_new))
    logging.info('base acc given new : {:.4f}'.format(va_base_given_new))
    logging.info('new acc given base : {:.4f}'.format(va_new_given_base))

    logs = dict(num_session=session + 1, acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new,
                new_acc_given_base=va_new_given_base)

    if session > 0:
        return vl, va, vacc, vprecision, vrecall, vf1, logs
    else:
        return vl, va, logs

import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def get_accuracy_confusion_matrix(model, testloader, num_classes, save_path):
    """
    计算模型在测试数据集上的准确率和混淆矩阵,并可视化混淆矩阵。
    
    参数:
    model (nn.Module): 要评估的深度学习模型
    testloader (DataLoader): 测试数据集的数据加载器
    num_classes (int): 分类问题的类别数量
    
    返回:
    accuracy (float): 模型在测试数据集上的准确率
    cm (numpy.ndarray): 模型的混淆矩阵
    """

    # 创建一个函数来格式化单元格值
    def format_cell(value):
        return f"{value:.1f}"

    # 设置模型为评估模式
    model.eval()
    
    # 初始化预测标签和真实标签列表
    y_true = []
    y_pred = []
    
    # 在测试数据集上进行预测
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images.to('cuda'))
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    
    # 计算准确率
    accuracy = 100 * (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).astype('int') # 归一化
    
    # 更改默认字体
    plt.rcParams['font.family'] = 'Ubuntu'  # 这里以 'SimHei' 字体为例
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(8, 8), dpi=800)
    # 可视化混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))
    disp.plot(cmap='Blues', ax=ax)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks(np.arange(len(disp.display_labels)))
    ax.set_yticks(np.arange(len(disp.display_labels)))
    ax.set_xticklabels(disp.display_labels)
    ax.set_yticklabels(disp.display_labels)
    ax.tick_params(axis='both', which='major', pad=10)
    plt.savefig(save_path)
    
    return accuracy, cm

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