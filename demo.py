import numpy as np
import time, cv2, torch, yaml, argparse, re
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.resNet_crnn as resNet_crnn
from easydict import EasyDict as edict

def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, required=True)
    parser.add_argument('--image_path', type=str, help='the path to your image', required=True)
    parser.add_argument('--checkpoint', type=str, help='the path to your checkpoints', required=True)
    parser.add_argument('--device', type=str, help='the device of running model\ne. g. cpu, cuda', default='cpu')

    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    img_h, img_w = img.shape

    img = cv2.resize(img, (0,0), fx=config.MODEL.IMAGE_SIZE.W / img_w, fy=config.MODEL.IMAGE_SIZE.H / img_h, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 1))

    img = img.astype(np.float32)
    img = (img/255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))

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

    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    print('pre run model')
    model.eval()
    with torch.no_grad():
        model(torch.randn((1, 1, config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W), device=device))
    print('pre run model done...')

    started = time.time()

    img = cv2.imread(args.image_path)
    img = cv2.imdecode(np.fromfile(args.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    recognition(config, img, model, converter, device)

    print('elapsed time: %.3fms' % ((time.time() - started) * 1000))
