import argparse, yaml, os, torch
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.resNet_crnn as resNet_crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
from visdom import Visdom

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def main():
    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    visTrain = Visdom(env='train', port=3000)
    assert visTrain.check_connection()

    writer_dict = {
        'writer': visTrain,
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = resNet_crnn.get_crnn(config)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss().to(device)

    optimizer = utils.get_optimizer(config, model)
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    best_acc = 0.5
    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        print('=================== resume model and opt ====================')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            lr_scheduler.step(last_epoch - 1)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            best_acc = checkpoint["best_acc"]
        else:
            model.load_state_dict(checkpoint)
    


    # model_info(model)
    train_dataset = get_dataset(config)(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    firstDrawLrGraph = True
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()

        writer_dict['writer'].line(X=torch.IntTensor([writer_dict['train_global_steps']]),
            Y=torch.Tensor([lr_scheduler.get_lr()[0]]),
            win='lr', update=None if firstDrawLrGraph else 'append', opts=dict(title='lr'))
        if firstDrawLrGraph:
            firstDrawLrGraph = False

        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    writer_dict['writer'].close()

if __name__ == '__main__':

    main()