import os, random, re, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gen labels")
    parser.add_argument('--imgs_dir', '-imgs_dir', help='the dir of train imgs', type=str, required=True)
    parser.add_argument('--test_imgs_dir', '-test_imgs_dir', help='the dir of test imgs', type=str, required=True)
    parser.add_argument('--ouput_dir', '-ouput_dir', help='the dir of out put label', type=str, default='label_ano')
    parser.add_argument('--rate', '-rate', help='the rate of train and valid', type=float, default=0.8)
    args = parser.parse_args()

    imgsDir = args.imgs_dir
    testImgsDir = args.test_imgs_dir
    ouputDir = args.ouput_dir
    train_rate = args.rate

    if not os.path.isdir(ouputDir):
        os.makedirs(ouputDir)
    
    imgsList = os.listdir(imgsDir)
    testFiles = os.listdir(testImgsDir)
    random.shuffle(imgsList)

    num = int(len(imgsList) * train_rate)
    trainFiles = imgsList[:num]
    valFiles = imgsList[num:]

    trainOut = open(os.path.join(ouputDir, 'train.txt'), 'w', encoding='utf8')
    valOut = open(os.path.join(ouputDir, 'val.txt'), 'w', encoding='utf8')
    testOut = open(os.path.join(ouputDir, 'test.txt'), 'w', encoding='utf8')

    if not trainOut.writable() or not valOut.writable() or not testOut.writable():
        print("fail to open files")
        exit(-1)

    for fileName in trainFiles:
        matchRes = re.match(r'^([A-Z0-9]+)(_?.*)\.(\w+)$', fileName)
        if matchRes is not None:
            trainOut.write(fileName + ' ' + matchRes.group(1) + '\n')
    trainOut.close()

    for fileName in valFiles:
        matchRes = re.match(r'^([A-Z0-9]+)(_?.*)\.(\w+)$', fileName)
        if matchRes is not None:
            valOut.write(fileName + ' ' + matchRes.group(1) + '\n')
    valOut.close()

    for fileName in testFiles:
        matchRes = re.match(r'^([A-Z0-9]+)\.(\w+)$', fileName)
        if matchRes is not None:
            testOut.write(fileName + ' ' + matchRes.group(1) + '\n')
    testOut.close()
    
    print('all done')