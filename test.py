import numpy as np, re
import time, cv2, torch, yaml, argparse, tqdm
import lib.utils.utils as utils
import lib.models.resNet_crnn as resNet_crnn
from easydict import EasyDict as edict
from lib.dataset import _OWN
from torch.utils.data import DataLoader

def parse_arg():
    parser = argparse.ArgumentParser(description="test")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, help='the path to your checkpoints', required=True)
    parser.add_argument('--device', type=str, help='the device of running model\ne. g. cpu, cuda', default='cpu')
    parser.add_argument('--image_dir', type=str, help='the path to your images dir', default='')
    parser.add_argument('--label_path', type=str, help='the path to your test label file', required=True)

    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def test(config, model, labelPath):
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    model.eval()
    model.to(device)

    with torch.no_grad():
        print('pre run....')
        model(torch.randn((1, 1, config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W)).to(device))
        print('done')

    test_dataset = _OWN(config, labelPath=labelPath)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    n_correct = 0
    showResList = []
    sum_time = 0
    with torch.no_grad():
        proc = tqdm.tqdm(total=len(test_dataset), unit='img')
        for inp, idx in test_loader:
            labels = utils.get_batch_label(test_dataset, idx)

            st = time.time()
            preds = model(inp.to(device)) # [imgW / 4 (seqLen), B, nClass]

            batch_size = inp.size(0)
            proc.update(batch_size)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)

            _, preds = preds.max(2) # [seqLen, B]
            preds = preds.transpose(1, 0).contiguous().view(-1) #[B, seqLen] -> [B * seqLen]
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            if not isinstance(sim_preds, list):
                sim_preds = [sim_preds]
            sum_time += time.time() - st
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1
                else:
                    showResList.append((pred, target))
        proc.close()

    n_test = len(test_dataset)
    print("[#correct:{} / #total:{}]".format(n_correct, n_test))
    accuracy = n_correct / float(n_test)
    print('accuray: {:.4f}'.format(accuracy))
    print('use %.3fms/img' % (float(sum_time) / n_test * 1000.0))

    if len(showResList) > 0:
        print('wrong imgs:')
        for pred, tg in showResList:
            print('%s\t%s' % (pred, tg))

if __name__ == '__main__':
    config, args = parse_arg()

    device = None
    if re.match(r'^(cpu)|(cuda):?\d?$', args.device) is not None:
        device = torch.device(args.device)
    else:
        print('the device is not right')
        exit(-1)

    model = resNet_crnn.get_crnn(config)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    config.DATASET.ROOT = args.image_dir
    test(config, model, args.label_path)
