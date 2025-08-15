from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import torch.optim
from torch.backends import cudnn
import torch.nn as nn
import random

from reid.datasets import get_data
from reid.utils.metrics import R1_mAP_eval
from reid.utils.logging import Logger
from reid.utils.serialization import save_checkpoint
from reid.utils.my_tools import *
from reid.models.vit_pytorch import build_vit_backbone
from reid.trainer import Trainer
from reid.utils.lr_scheduler import create_scheduler
from reid.utils.make_optimizer import make_optimizer
from config import cfg
def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    print('cuda:', torch.cuda.is_available())
    main_worker(args)


def initclassifier(model, num_class=[], data="market", init_loader=None):
    if data == "cuhksysu":
        org_classifier1_params = model.classifier1.weight.data
        model.classifier1 = nn.Linear(768, num_class[0] + num_class[1], bias=False)
        model.cuda()
        model.classifier1.weight.data[:(num_class[0])].copy_(org_classifier1_params)
        # Initialize classifer with class centers
        class_centers = initial_classifier(model, init_loader)
        model.classifier1.weight.data[num_class[0]:].copy_(class_centers)
        model.cuda()

    if data == "dukemtmc":
        org_classifier1_params = model.classifier1.weight.data
        model.classifier1 = nn.Linear(768, num_class[0] + num_class[1] + num_class[2], bias=False)
        model.cuda()
        model.classifier1.weight.data[:(num_class[0] + num_class[1])].copy_(org_classifier1_params)
        # Initialize classifer with class centers
        class_centers = initial_classifier(model, init_loader)
        model.classifier1.weight.data[num_class[0] + num_class[1]:].copy_(class_centers)
        model.cuda()
    if data == "msmt17":
        org_classifier1_params = model.classifier1.weight.data
        model.classifier1 = nn.Linear(768, num_class[0] + num_class[1] + num_class[2] + num_class[3], bias=False)
        model.cuda()
        model.classifier1.weight.data[:(num_class[0] + num_class[1] + num_class[2])].copy_(org_classifier1_params)
        # Initialize classifer with class centers
        class_centers = initial_classifier(model, init_loader)
        model.classifier1.weight.data[num_class[0] + num_class[1] + num_class[2]:].copy_(class_centers)
        model.cuda()

    if data == "CUHK03":
        org_classifier1_params = model.classifier1.weight.data
        model.classifier1 = nn.Linear(768, num_class[0] + num_class[1] + num_class[2] + num_class[3] + num_class[4], bias=False)
        model.cuda()
        model.classifier1.weight.data[:(num_class[0] + num_class[1] + num_class[2] + num_class[3])].copy_(org_classifier1_params)
        # Initialize classifer with class centers
        class_centers = initial_classifier(model, init_loader)
        model.classifier1.weight.data[num_class[0] + num_class[1] + num_class[2] + num_class[3]:].copy_(class_centers)
        model.cuda()

def main_worker(args):
    cudnn.benchmark = True
    log_name = 'train.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

   # read parameters
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    # Create data loaders  order1
    dataset_market, num_classes_market, train_loader_market, test_loader_market, _ = \
        get_data('market1501', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)
    dataset_cuhksysu, num_classes_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk_sysu', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_dukemtmc, num_classes_dukemtmc, train_loader_dukemtmc, test_loader_dukemtmc, init_loader_dukemtmc = \
        get_data('dukemtmc', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_msmt17, num_classes_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cusk03, num_classes_cuhk03, train_loader_cuhk03, test_loader_cuhk03, init_loader_cuhk03 = \
        get_data('CUHK03', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    print(num_classes_market, num_classes_cuhksysu, num_classes_dukemtmc, num_classes_msmt17, num_classes_cuhk03)

    # Create model
    model = build_vit_backbone(num_classes_market, cfg)
    model.cuda()

    # Market Evaluator
    start_epoch = 0
    evaluators = [R1_mAP_eval(len(dataset_market.query), max_rank=50, feat_norm=True)]
    names = ['market1501']
    test_loaders = [test_loader_market]

    # initialize Opitimizer and lr
    optimizer = make_optimizer(args, model)
    lr_scheduler = create_scheduler(optimizer=optimizer, epochs=60, lr=args.lr) #

    # Start training
    print('Continual training starts!')

    # Market1501 start training
    trainer = Trainer(args, model=model, tmodel=None, optimizer=optimizer, num_classes=num_classes_market,
                 data_loader_train=train_loader_market, data_loader_replay=None, training_phase=1, replay=False, margin=args.margin)
    for epoch in range(start_epoch, args.epochs):
        train_loader_market.new_epoch()
        trainer.train(epoch)
        lr_scheduler.step(epoch)

        if (epoch == args.epochs - 1):
            print(".............")
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, True, fpath=osp.join(args.logs_dir, 'current_checkpoint_step_1.pth.tar'))
            print(".............")
            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                eval_func(epoch, evaluator, model, test_loader, name)
            print(".............")
            print('------Market-1501 Finished Training------')

    # Select replay data of market1501
    replay_dataloader, market_replay_dataset = select_replay_samples(model, dataset_market, training_phase=1)

    # Expand the dimension of classifier
    initclassifier(model, num_class=[num_classes_market, num_classes_cuhksysu], data="cuhksysu",
                   init_loader=init_loader_chuksysu)
    #creat old model and Expand the dimension of classifier
    old_model = build_vit_backbone(num_classes_market, cfg)
    initclassifier(old_model, num_class=[num_classes_market, num_classes_cuhksysu], data="cuhksysu",
                   init_loader=init_loader_chuksysu)
    add_num = num_classes_market

    # Create old frozen model
    tmp_state_dict = old_model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = model.state_dict()[k]
    old_model.load_state_dict(tmp_state_dict)

    num_query = len(dataset_cuhksysu.query)
    # Re-initialize optimizer and lr
    optimizer = make_optimizer(args, model)
    lr_scheduler = create_scheduler(optimizer=optimizer, epochs=args.epochs, lr=args.lr)

    # Market1501 start training
    trainer = Trainer(args, model=model, tmodel=old_model, optimizer=optimizer,
                      num_classes=num_classes_cuhksysu + num_classes_market,
                      data_loader_train=train_loader_cuhksysu, data_loader_replay=replay_dataloader,
                      training_phase=2, add_num=add_num, replay=True, margin=args.margin)
    for epoch in range(start_epoch, args.epochs):

        train_loader_cuhksysu.new_epoch()
        trainer.train(epoch)
        lr_scheduler.step(epoch)

        if (epoch == args.epochs - 1):

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, True, fpath=osp.join(args.logs_dir, 'current_checkpoint_step_2.pth.tar'))

            evaluators.append(R1_mAP_eval(num_query, max_rank=50, feat_norm=True))
            names.append('cuhksysu')
            test_loaders.append(test_loader_cuhksysu)

            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                eval_func(epoch, evaluator, model, test_loader, name)
            print('------Cuhksysu Finished Training------')

    # Select replay data of cuhksysu
    replay_dataloader, cuhksysu_replay_dataset = select_replay_samples(model,
    dataset_cuhksysu, training_phase=2, add_num=num_classes_market, old_datas=market_replay_dataset)

    # Expand the dimension of classifier
    initclassifier(model, num_class=[num_classes_market, num_classes_cuhksysu, num_classes_dukemtmc], data="dukemtmc",
                   init_loader=init_loader_dukemtmc)
    initclassifier(old_model, num_class=[num_classes_market, num_classes_cuhksysu, num_classes_dukemtmc], data="dukemtmc",
                   init_loader=init_loader_dukemtmc)

    add_num = num_classes_cuhksysu + num_classes_market

    # Create old frozen model
    tmp_state_dict = old_model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = model.state_dict()[k]
    old_model.load_state_dict(tmp_state_dict)

    # Re-initialize optimizer and lr
    optimizer = make_optimizer(args, model)
    lr_scheduler = create_scheduler(optimizer=optimizer, epochs=args.epochs, lr=args.lr)

    # Dukemtmc start training
    trainer = Trainer(args, model=model, tmodel=old_model, optimizer=optimizer,
                      num_classes=num_classes_dukemtmc + add_num,
                      data_loader_train=train_loader_dukemtmc, data_loader_replay=replay_dataloader,
                      training_phase=3, add_num=add_num, replay=True, margin=args.margin)

    for epoch in range(start_epoch, args.epochs):

        train_loader_dukemtmc.new_epoch()
        trainer.train(epoch)
        lr_scheduler.step(epoch)

        if (epoch == args.epochs - 1):
            if epoch == args.epochs - 1:
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                }, True, fpath=osp.join(args.logs_dir, 'current_checkpoint_step_3.pth.tar'))

            test_loaders.append(test_loader_dukemtmc)
            evaluators.append(R1_mAP_eval(len(dataset_dukemtmc.query), max_rank=50, feat_norm=True))
            names.append('dukemtmc')

            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                eval_func(epoch, evaluator, model, test_loader, name)
            print('------DukeMTMC Finished Training------')

    #select dukemtmc replay
    replay_dataloader, dukemtmc_replay_dataset = select_replay_samples(model,
    dataset_dukemtmc, training_phase=3, add_num=add_num, old_datas=cuhksysu_replay_dataset)

    initclassifier(model,
                   num_class=[num_classes_market, num_classes_cuhksysu, num_classes_dukemtmc, num_classes_msmt17],
                   data="msmt17", init_loader=init_loader_msmt17)
    initclassifier(old_model,
                   num_class=[num_classes_market, num_classes_cuhksysu, num_classes_dukemtmc, num_classes_msmt17],
                   data="msmt17", init_loader=init_loader_msmt17)
    add_num = num_classes_cuhksysu + num_classes_market + num_classes_dukemtmc

    # Create old frozen model
    tmp_state_dict = old_model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = model.state_dict()[k]
    old_model.load_state_dict(tmp_state_dict)

    # Re-initialize optimizer and lr
    optimizer = make_optimizer(args, model)
    lr_scheduler = create_scheduler(optimizer=optimizer, epochs=args.epochs, lr=args.lr)

    # MSMT17 start training
    trainer = Trainer(args, model=model, tmodel=old_model, optimizer=optimizer,
                      num_classes=num_classes_msmt17 + add_num,
                      data_loader_train=train_loader_msmt17, data_loader_replay=replay_dataloader,
                      training_phase=4, add_num=add_num, replay=True, margin=args.margin)

    for epoch in range(start_epoch, args.epochs):
        train_loader_msmt17.new_epoch()
        trainer.train(epoch)
        lr_scheduler.step(epoch)

        if epoch == args.epochs - 1:

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, True, fpath=osp.join(args.logs_dir, 'current_checkpoint_step_4.pth.tar'))

            evaluators.append(R1_mAP_eval(len(dataset_msmt17.query), max_rank=50, feat_norm=True))
            names.append("msmt17")
            test_loaders.append(test_loader_msmt17)

            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                eval_func(epoch, evaluator, model, test_loader, name)
            print('------MSMT17 Finished Training------')

    # Select replay data of MSMT17
    replay_dataloader, msmt17_replay_dataset = select_replay_samples(model,
    dataset_msmt17, training_phase=4, add_num=add_num, old_datas=dukemtmc_replay_dataset)

    initclassifier(model,
                   num_class=[num_classes_market, num_classes_cuhksysu, num_classes_dukemtmc, num_classes_msmt17,
                              num_classes_cuhk03], data="CUHK03", init_loader=init_loader_cuhk03)
    initclassifier(old_model,
                   num_class=[num_classes_market, num_classes_cuhksysu, num_classes_dukemtmc, num_classes_msmt17,
                              num_classes_cuhk03], data="CUHK03", init_loader=init_loader_cuhk03)
    add_num = num_classes_dukemtmc + num_classes_market + num_classes_cuhksysu + num_classes_msmt17

    # Create old frozen model
    tmp_state_dict = old_model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = model.state_dict()[k]
    old_model.load_state_dict(tmp_state_dict)

    # Re-initialize optimizer and lr
    optimizer = make_optimizer(args, model)
    lr_scheduler = create_scheduler(optimizer=optimizer, epochs=args.epochs, lr=args.lr)
    trainer = Trainer(args, model=model, tmodel=old_model, optimizer=optimizer,
                      num_classes=num_classes_cuhk03 + add_num,
                      data_loader_train=train_loader_cuhk03, data_loader_replay=replay_dataloader,
                      training_phase=5, add_num=add_num, replay=True, margin=args.margin)
    # CUHK03 start training
    for epoch in range(start_epoch, args.epochs):
        train_loader_cuhk03.new_epoch()
        trainer.train(epoch)
        lr_scheduler.step(epoch)

        if epoch == args.epochs - 1:

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, True, fpath=osp.join(args.logs_dir, 'current_checkpoint_step_5.pth.tar'))

            evaluators.append(R1_mAP_eval(len(dataset_cusk03.query), max_rank=50, feat_norm=True))
            names.append("CUHKO3")
            test_loaders.append(test_loader_cuhk03)

            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                eval_func(epoch, evaluator, model, test_loader, name)
            print('------CUHK03 Finished Training------')

    print('finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument(
        "--config_file", default="configs/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-br', '--replay-batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--lr', type=float, default=0.000005,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20, 40],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join('***'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument("--device_id", default=[0, 1], type=int)
    main()
