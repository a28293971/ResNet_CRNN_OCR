# ResNet_CRNN_OCR
-------------------------------

## Introduction
This repo can use to train a OCR model which can recognize variable length character(depend on the width of img), and transform it's pytorch model to a jit script model, that can use libtorch to load and run it. This means it can be used in a production environment.

## Notice
This project refers to other people's code, thanks to their open source behavior to promote the development of this industry.  

+ the code of training and demo is based on [CRNN_Chinese_Characters_Rec](https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec)  
+ the model is based on [Resnet.CRNN](https://github.com/zamling/Resnet.CRNN)
+ the method of constructing dataset is based on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## Environment

The project has only been tested in the following environments:
+ OS: Windwos 10
+ CPU: i9-7900X, i7-7700
+ GPU: GTX1080, GTX2080Ti
+ CUDA: 10.2, 10.1
+ Pytorch: 1.7.1
+ Libtorch: 1.7.1
+ IED: VS Code, Visual Studio 2019, 2017
+ C++ Standard: C++14, C++17
+ OpenCV: 4.2, 4.4

## Dependencies

#### Python
+ Pytorch
+ OpenCV
+ visdom
+ tqdm
+ easydict
#### Cpp
+ libtorch
+ OpenCV

## Train
1. Prepare the train and test dataset.(*the model only support the img with a height of 32 or you can modify "lib\models\resNet_crnn.py"*)
2. run `python gen_labels.py -imgs_dir PATH/TO/YOUR/RAIN/DATASET -test_imgs_dir PATH/TO/YOUR/TEST/DATASET`, and it will generate 3 files(train.txt, val.txt and test.txt) in the dir **label_ano**. (*the img file name must be "label.png[tif, jpg]"(e.g. 37GD28d.png). or modify the file "gen_labels.py"*)
3. copy **lib/config/OWN_config.yaml** and rename it to your own name.
4. modify "YOUR/OWN/CONFIG.yaml" to adapt the situation you want to train. Especially, change **DATASET.ROOT** to your train imgs dir, and **DATASET.JSON_FILE** to the file "train.txt" and "val.txt" that before generate. Fill all characters you wanna train in **DATASET.ALPHABETS**.
5. run `python train.py --cfg PATH/TO/YOUR/CONFIG.yaml`

## Test
run `python test.py --cfg PATH/TO/YOUR/CONFIG.yaml --checkpoint PATH/TO/YOUR/MODE/LFILE --device CPU[CUDA[:0,1]] --image_dir PATH/TO/YOUR/IMGS/DIR --label_path PATH/TO/YOUR/LABEL/FILE`. This prog will show the accuracy of your test dataset and time used per picture.

## Demo
run `python demo.py --cfg PATH/TO/YOUR/CONFIG.yaml --image_path PATH/TO/YOUR/IMG --checkpoint PATH/TO/YOUR/MODEL/FILE --device CPU[CUDA[:0,1]]`, and will see the predict result and time used.

## C++ use
This project also provides a solution to export the model, and test it.

### export
run `python exportCppModule.py --mode export --cfg PATH/TO/YOUR/CONFIG.yaml --model_path PATH/TO/YOUR/PYTORCH/MODEL/FILE --device CPU[CUDA[:0,1]] --output_path PATH/TO/EXPORT/TORCH/SCRIPT/MODEL`, this cmd will export the original pytorch model to a torch script model, that it can run on C++ and Python platform. **Note: Up to now, there are problems with pytorch RNN model when export to a script model. It manifests as that, the weight matrix of RNN can not be move to other device. So, if export a CPU script model and run it in GPU, it will occur an error like 'not at the same device'. Even, export a CUDA:0 model, and run on CUDA:1, it also dosen't work. Overall, if you when to use torch script model contains RNN, I recommend keeping it on the same device.**

### test
run `python exportCppModule.py --mode test --cfg PATH/TO/YOUR/CONFIG.yaml --model_path PATH/TO/YOUR/TORCH/SCRIPT/MODEL/FILE --device CPU[CUDA[:0,1]] --image_path PATH/TO/YOUR/IMAGE`. it will show the prediction result and time used with the model. **Note: As mentioned above, keeping it on the same device.**

### run in C++
The demo file is in 'cppSource' folder.  
Just need to change some const variable,which are top of 'main()' function, and ensure that the environment has been configured right. Compile the C++ source code, and run the file, there will be some result.