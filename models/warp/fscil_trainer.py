from base import Trainer
import os.path as osp
import torch.nn as nn
import copy
from copy import deepcopy
import pandas as pd
from os.path import exists as is_exists

from .helper import *
from utils import *
from dataloader.data_utils import *
from models.switch_module import switch_module
# from tsne_torch import plot_sne


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_up_model()

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.val_model = MYNET(self.args, mode=self.args.base_mode)

        # GPU并行化处理
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.val_model = nn.DataParallel(self.val_model, list(range(self.args.num_gpu)))
        self.val_model = self.val_model.cuda()

        # 加载模型参数
        if self.args.model_dir is not None:
            logging.info('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']

        else:
            logging.info('random init params')
            if self.args.start_session > 0:
                logging.info('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())


    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        # 对于每个任务
        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = get_dataloader(args, session)

            if args.epochs_base > 0 or session == 0:
                self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label
                logging.info(f'new classes for this session:\n {np.unique(train_set.targets)}')
                optimizer, scheduler = get_optimizer(args, self.model)

                if args.epochs_base == 0:
                    if 'ft' in args.new_mode:
                        self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args) # 将分类器原型替换为平均特征向量
                        self.model.module.mode = args.new_mode
                        self.val_model.load_state_dict(deepcopy(self.model.state_dict()), strict=False)
                        self.val_model.module.mode = args.new_mode
                        tsl, tsa, logs = test(self.val_model, testloader, args.epochs_base, args, session) # 测试基类的性能
                        switch_module(self.model) # 替换特定的卷积层
                        compute_orthonormal(args, self.model, train_set) # 计算新的正交基底
                        identify_importance(args, self.model, train_set, keep_ratio=args.fraction_to_keep)
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    else:
                        self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                        self.model.module.mode = args.new_mode
                        tsl, tsa, logs = test(self.model, testloader, args.epochs_base, args, session)
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))


                else:
                    for epoch in range(args.epochs_base):
                        start_time = time.time()
                        # train base sess
                        tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                        # test model with all seen class
                        tsl, tsa, logs = test(self.model, testloader, epoch, args, session)

                        # Note that, although this code evaluates the test accuracy and save the max accuracy model,
                        # we do not use this model. We use the "last epoch" pretrained model for incremental sessions.
                        if (tsa * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            logging.info('********A better model is found!!**********')
                            logging.info('Saving model to :%s' % save_model_dir)
                        logging.info('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                           self.trlog['max_acc'][session]))

                        self.trlog['train_loss'].append(tl)
                        self.trlog['train_acc'].append(ta)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]
                        result_list.append(
                            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                        print('This epoch takes %d seconds' % (time.time() - start_time),
                              '\nstill need around %.2f mins to finish this session' % (
                                      (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                        scheduler.step()

                    result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                    save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_last_epoch.pth')
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)

                    # save the last epoch model here
                    self.best_model_dict = deepcopy(self.model.state_dict())

                    if not args.not_data_init:
                        self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                        self.model.load_state_dict(self.best_model_dict)
                        self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                        best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc_replace_head.pth')
                        logging.info('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        torch.save(dict(params=self.model.state_dict()), best_model_dir)

                        self.model.module.mode = 'avg_cos'
                        tsl, tsa, logs = test(self.model, testloader, 0, args, session)
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        logging.info('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                logging.info("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session) # 在分类层中添加新类的参数

                if 'ft' in args.new_mode:
                    restore_weight(self.model)  # 恢复特征空间
                    self.val_model.load_state_dict(deepcopy(self.model.state_dict()), strict=False)
                    tsl, tsa, logs = test(self.val_model, testloader, 0, args, session)
                else:
                    tsl, tsa, logs = test(self.model, testloader, 0, args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                self.best_model_dict = deepcopy(self.model.state_dict())
                logging.info('Saving model to :%s' % save_model_dir)
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                logging.info('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

                # if session > 0:
                #     compute_orthonormal(args, self.model, train_set) # 计算新的正交基底

        embedding_list, label_list = get_features(testloader, testloader.dataset.transform, self.model)
        save_s_tne(embedding_list.numpy(), label_list.numpy(), args.save_path)

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        logging.info(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        logging.info('Base Session Best epoch: %s' % self.trlog['max_acc_epoch'])
        logging.info('Total time used %.2f mins' % total_time)

