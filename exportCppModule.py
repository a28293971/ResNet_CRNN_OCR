import torch, time, re, os, argparse, yaml
from torchvision import transforms
import lib.models.resNet_crnn as resNet_crnn
import cv2 as cv
from easydict import EasyDict as edict

def parse_arg():
    parser = argparse.ArgumentParser(description="export model")
    parser.add_argument('--mode', help='export model or test script model', type=str, default='export', choices=['export', 'test'])
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='if export mode, it means the input pytorch model path. \
        if test mode, it means the input torch script model path')
    parser.add_argument('--device', type=str, default='cpu', 
        help='the device of running model.e. g. cpu, cuda. This option depend on what mode you use, when export mode, it means the \
        device of model export to. If test mode, it means device of model run to.')
    parser.add_argument('--output_path', help='the output model path', type=str, default='traecd_model.pt')
    parser.add_argument('--image_path', help='path to img for testing model', type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def decode(singleTensor: torch.Tensor, alphabets: str) -> str:
    assert singleTensor.dim() == 1, "the dim of tensor must be 1"
    return ''.join([alphabets[singleTensor[i] - 1] 
            for i in range(singleTensor.size(0))
            if singleTensor[i] != 0 and not (i > 0 and singleTensor[i - 1] == singleTensor[i])])

def testModel(config, args, device):
    model = torch.jit.load(args.model_path, map_location=device)
    model.eval()

    img = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)
    label = re.match(r'^([A-Z0-9]+)(_?.*)\.(.+?)$', os.path.basename(args.image_path)).group(1)

    func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W)),
        transforms.Normalize(mean=[config.DATASET.MEAN],
                            std=[config.DATASET.STD])
    ])

    # init run
    with torch.no_grad():
        model(torch.randn((1, 1, config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W), device=device))

    st = time.time()
    inputImg = func(img)

    with torch.no_grad():
        inputImg.unsqueeze_(0)
        output = model(inputImg)

    _, preds = torch.max(output, 2)
    preds = preds.permute([1, 0])
    alphabets =  config.DATASET.ALPHABETS + '-'
    decodeRes = decode(preds[0], alphabets)
    print('run in %s\ntotal used %.3fms' % (args.device, (time.time() - st) * 1000))

    print('the pred res is %s' % 'right' if decodeRes == label else 'wrong')
    print('res: %s\nreal label: %s' % (decodeRes, label))

def exptModel(config, args, device):
    model = resNet_crnn.get_crnn(config)
    print('loading pretrained model from {0}'.format(args.model_path))
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    traced_script_module = torch.jit.trace(model, torch.randn((1, 1, config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W), device=device))
    traced_script_module.save(args.output_path)

    print('trace done!\n had save to', args.output_path, 'in device:', args.device)

if __name__ == "__main__":
    config, args = parse_arg()

    if args.mode not in ['export', 'test']:
        print('the mode must be export or test')
        exit(-1)
    
    device = None
    if re.match(r'^(cpu)|(cuda):?\d?$', args.device) is not None:
        device = torch.device(args.device)
    else:
        print('the device is not right')
        exit(-1)

    if args.mode == 'export':
        exptModel(config, args, device)
    else:
        if args.image_path is None:
            print('it need an img when run test mode')
            exit(-1)
        testModel(config, args, device)
