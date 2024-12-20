from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.distributed as dist
import _init_paths

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import json
import torch.utils.data
from torchvision.transforms import transforms as T
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory
import lib.utils.misc as utils
from torch.utils.data.distributed import DistributedSampler
# from mmcv.runner import init_dist

def run(opt):
    utils.init_distributed_mode(opt)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task, opt.multi_scale)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    print("Dataset root: %s" % dataset_root)
    f.close()

    transforms = T.Compose([T.ToTensor()])

    dataset = Dataset(opt, dataset_root, trainset_paths, img_size=opt.input_wh, augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print("opt:\n", opt)
    logger = Logger(opt)
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    device = torch.device('cuda')
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Image data transformations
    sampler_train = DistributedSampler(dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, opt.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_sampler=batch_sampler_train,
                                   pin_memory=True)

    logger = Logger(opt)
   
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu],find_unused_parameters=True)
    model_without_ddp = model.module
    start_epoch = 0
    # Get dataloader
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model_without_ddp,
                                                   opt.load_model,
                                                   optimizer,
                                                   opt.resume,
                                                   opt.lr,
                                                   opt.lr_step)
    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    if utils.is_main_process():
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(0)),
                       0, model, optimizer)
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        sampler_train.set_epoch(epoch)
        # Train an epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)
    
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            if utils.is_main_process():
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:  # mcmot_last_track or mcmot_last_det
            if opt.id_weight > 0:  # do tracking(detection and re-id)
                if utils.is_main_process():
                    save_model(os.path.join(opt.save_dir, 'mcmot_last_track_' + opt.arch + '.pth'),
                           epoch, model, optimizer)
            else:  # only do detection
                if utils.is_main_process():
                    save_model(os.path.join(opt.save_dir, 'mcmot_last_det_' + opt.arch + '.pth'),
                           epoch, model, optimizer)
        logger.write('\n')

        if epoch in opt.lr_step:
            if utils.is_main_process():
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 5 == 0:
            if utils.is_main_process():
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    print(opt)
    run(opt)
