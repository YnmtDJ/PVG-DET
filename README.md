# GCN_DETECTION
GCN for Object Detection

# Requirements
numpy 1.24.3  
tqdm 4.65.0  
opencv-contrib-python 4.8.1.78  
scipy 1.11.4  
torch 2.1.0  
torchvision 0.16.0
pycocotools 2.0.7  

## Dataset
COCO  
PASCAL VOC  
ImageNet  
Objects365

[VisDrone](https://github.com/VisDrone/VisDrone-Dataset) Car, Person 1.5G image_size: (540, 960) (765, 1360) (1080, 1920) (1060, 1400)  
[DOTA](https://captain-whu.github.io/DOTA/dataset.html) Ariel (high resolution) 20G  
[xview](http://xviewdataset.org/) overhead imagery 33G  
[iSAID](https://captain-whu.github.io/iSAID/dataset.html) Ariel Instance Segmentation
[HRSC2016](http://www.escience.cn/people/liuzikun/DataSet.html) Ship (high resolution) 4GB  
[Tsinghua-Tencent 100k](https://cg.cs.tsinghua.edu.cn/traffic-sign/)  Traffic-Sign 23G  
[AI-TOD](http://m6z.cn/5MjlYk) Ariel (high resolution) 22.95G  
[TinyPerson](http://m6z.cn/6vqF3T) Person 8.23G  
[CityPersons](https://www.cityscapes-dataset.com/) Person  
[Spotting Birds](https://github.com/IIM-TTIJ/MVA2023SmallObjectDetection4SpottingBirds) 10G  

## Model Setting
COCO: [16, 32, 64, 128, 256], (400, 667), [96, 160, 416, 640], [2, 2, 6, 2], 96  
VisDrone: [8, 16, 32, 64, 128], (800, 1333), [96, 160, 416, 640], [2, 2, 6, 2], 96  

## TODO
norm, experiment
