from .MultiCenterLoss import MultiCenterLoss
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def base_train(model, trainloader, optimizer, scheduler, epoch, args, fc, centers):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    if epoch == args.knn_epoch:
        # model.mode = 'ft_cos'
        # for module in model.modules():
        #     if hasattr(module, 'weight') and not isinstance(module, WaRPModule):
        #         for param in module.parameters():
        #             param.requires_grad = False
        avg_features = torch.tensor(extract_features_and_cluster(trainloader, model, 0, args.multi_proto_num), dtype=torch.float32, device='cuda')
        with torch.no_grad():
            model.centers.weight.copy_(avg_features)
            model.centers.weight.requires_grad_(True)
        model.is_multi = True
    print(model.centers.weight)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        
        if epoch < args.knn_epoch:
            x_feature, logits = model(data)
            logits = logits[:, :args.base_class]
            total_loss = F.cross_entropy(logits, train_label)
        else:
            x_feature, logits = model(data, is_multi=True)
            logits = logits[:, :args.base_class]
            multi_center_loss = MultiCenterLoss(target_class=0, model=model)
            weight = torch.ones(args.base_class).cuda()
            # weight[0] = 0.1
            loss = F.cross_entropy(logits, train_label, weight=weight)
            total_loss = loss + args.alpha1 * multi_center_loss(x_feature, train_label)

        acc = count_acc(logits, train_label)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        # if model.is_multi == True:
        #     with torch.no_grad():
        #         model.centers.weight -= lrc * model.centers.weight.grad
        #     model.centers.weight.grad.zero_()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def extract_features_and_cluster(trainloader, model, target_class, k_clusters):
    # Step 1: Gather features for the specified class
    features = []
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            # Only select samples from the target class
            mask = labels == target_class
            if torch.any(mask):
                # Extract features for the target class samples
                target_inputs = inputs[mask]
                target_features, logits = model(target_inputs)
                target_features = target_features.cpu().numpy()
                features.append(target_features)
    
    # Flatten list of features arrays to single array
    features = np.concatenate(features, axis=0)
    # features_normalized = normalize(features, norm='l2', axis=1)

    # Step 2: Apply KMeans clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(features)
    cluster_labels = kmeans.labels_

    # Step 3: Calculate average feature for each cluster
    avg_features = []
    for cluster_id in range(k_clusters):
        cluster_features = features[cluster_labels == cluster_id]
        avg_features.append(np.mean(cluster_features, axis=0))
    
    return avg_features

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
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

    model.fc.weight.data[:args.base_class] = proto_list

    return model

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

            _, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(num_classes):
                class_mask = (labels == i)
                correct_per_class[i] += (preds[class_mask] == labels[class_mask]).sum().item()
                total_per_class[i] += class_mask.sum().item()

    acc_per_class = [correct / total if total != 0 else 0 for correct, total in zip(correct_per_class, total_per_class)]
    return acc_per_class

import numpy as np
import matplotlib.pyplot as plt
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
            _, outputs = model(images.to('cuda'))
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


def test(model, testloader, epoch, args, session, result_list=None, centers=None):
    test_class = args.base_class + session * args.way
    logging.info(get_accuracy_per_class(model, testloader, test_class))
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
    va5= Averager()
    vacc = Averager()  # 准确率平均
    vprecision = Averager()  # 精确率平均
    vrecall = Averager()  # 召回率平均
    vf1 = Averager()  # F1平均
    lgt=torch.tensor([])  # logits
    lbs=torch.tensor([])  # labels
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            if epoch < args.knn_epoch:
                x_feature, logits = model(data)
            else:
                x_feature, logits = model(data, is_multi=True)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

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

            # logging.info(labels)
            # logging.info(preds)
            vacc.add(accuracy_score(labels, preds))
            vprecision.add(precision_score(labels, preds, average='macro', zero_division=0))
            vrecall.add(recall_score(labels, preds, average='macro', zero_division=0))
            vf1.add(f1_score(labels, preds, average='macro', zero_division=0))

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt=torch.cat([lgt,logits.cpu()])
            lbs=torch.cat([lbs,test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5= va5.item()
        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
        vacc = vacc.item()
        vprecision = vprecision.item()
        vrecall = vrecall.item()
        vf1 = vf1.item()
        
        logging.info('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}, accuracy={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}'.format(epoch, vl, va, va5, vacc, vprecision, vrecall, vf1))

        lgt=lgt.view(-1, test_class)
        lbs=lbs.view(-1)

        if session > 0:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt,lbs)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            
            result_list.append(f"Seen Acc:{va_base_given_new}  Unseen Acc:{va_new_given_base}")
            return vl, (va_base_given_new, va_new_given_base, va, vacc, vprecision, vrecall, vf1)
        else:
            return vl, va
