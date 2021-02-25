# Detection in Crowded Scenes: One Proposal, Multiple Predictions

This is the pytorch re-implementation of the paper "[Detection in Crowded Scenes: One Proposal, Multiple Predictions](https://arxiv.org/abs/2003.09163)" that published in CVPR 2020.

<!-- Our method aiming at detecting highly-overlapped instances in crowded scenes. -->
Object detection in crowded scenes is challenging. When objects gather, they tend to overlap largely with each other, leading to occlusions. Occlusion caused by objects of the same class is called intra-class occlusion, also referred to as crowd occlusion. Object detectors need to determine the locations of different objects in the crowd and accurately delineate their boundaries. Many cases are quite challenging even for human annotators.

To address the aforementioned problem, this paper proposed a schema that one anchor/proposal can predict multiple predictions simultaneously. With this scheme, the predictions of nearby proposals are expected to infer the **same set** of instances, rather than **distinguishing individuals**, which is much easy for the model to learn. Besides, A new NMS method called set NMS is designed to remove the duplicates during the inference time. The EMD loss is devised to obtain the minimal loss during optimization based on the truth that a set of combinations can be obtained between the predictions and groundtruth boxes. Therefore, the combination that produces the minimal loss can be chosen to better optimize the model during training. Additionally, the proposed schema can be deployed on the mainstream detectors such as [Cascade RCNN](https://arxiv.org/pdf/1712.00726.pdf), [FPN](https://arxiv.org/pdf/1612.03144.pdf) and also one-stage detector [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf). The implementation details can be viewed in the repository.


The model structure and results are shown here:

<img width=60% src="https://github.com/Purkialo/images/blob/master/CrowdDet_arch.jpg"/>
<img width=90% src="https://github.com/Purkialo/images/blob/master/CrowdDet_demo.jpg"/>

# Citation

If you use the code in your research, please cite:
```
@InProceedings{Chu_2020_CVPR,
author = {Chu, Xuangeng and Zheng, Anlin and Zhang, Xiangyu and Sun, Jian},
title = {Detection in Crowded Scenes: One Proposal, Multiple Predictions},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

# Run
1. Requirements:
    * python 3.6.8, pytorch 1.5.0, torchvision 0.6.0, cuda 10.1

2. CrowdHuman data:
    * CrowdHuman is a benchmark dataset containing highly overlapped objects to better evaluate whether a detector can better handle crowd scenarios. The dataset can be downloaded from http://www.crowdhuman.org/. The path of the dataset is set in `config.py`.

3. Steps to run:
    * Step1:  training. More training and testing settings can be set in `config.py`.
    ```
    cd ROOT_DIR/model/DETECTOR_NAME/OWNER_NAME/project
    ```
    
    * Step2:  testing. If you have four GPUs, you can use ` -d 0-NUM_GPUS ` to use all of your GPUs.
              NUM_GPUS is the number of GPUs you would lik to use during inference,
              The result json file will be saved in the corresponding directory automatically.
    ```
    cd ROOT_DIR/model/DETECTOR_NAME/OWNER_NAME/project
    python3 test_net.py -d 0-NUM_GPUS -r 40 -e 50
    ```
    
    * Step3:  evaluating json, inference one picture and visulization json file. All of the value correpsponding the different evalutation metric will be calculated and be saved in a log file
    ```
    cd ROOT_DIR/model/DETECTOR_NAME/OWNER_NAME/project
    python3 demo.py
    ```

# Models

This proiect is a re-implementation based on Megengine.
We use pre-trained model from [MegEngine Model Hub](https://megengine.org.cn/model-hub). You can get this model from [GoogleDrive]().

All models are re-trained based on [ResNet-50](https://arxiv.org/pdf/1512.03385.pdf).

| | mAP | mMR | mJI | Model
| --- | --- | --- | --- | --- |
| RCNN FPN Baseline | 0.8728 | 0.4397 | 0.7915 | [rcnn_fpn_baseline.mge](https://drive.google.com/file/d/1IPVkS-wJ9e1r-x1tASJ6jfWcT7n1fOJr/view?usp=sharing) |
| RCNN EMD Simple   | 0.9002 | 0.4262 | 0.8217 | [rcnn_emd_simple.mge](https://drive.google.com/file/d/1rQrutc8GSawL7L2HWIv8BHd5VYy0WZFH/view?usp=sharing) |
<!-- | RCNN EMD with RM  | 0. | 0. | 0. | [rcnn_emd_refine.mge]() | -->
| Cascade FPN RCNN Baseline    | 0.8729 | 0.4264 | 0.8007 | [cascade_rcnn_fpn_baseline.mge](https://drive.google.com/file/d/1cFvP-bQRj09jdufkIwCNFVYZXej5dBX9/view?usp=sharing) |
| Cascade FPN RCNN EMD Simple  | 0.8990 | 0.4065 | 0.8247 | [cascade_rcnn_emd_simple.mge](https://drive.google.com/file/d/11V0V220gQ6sEjwiZracBodMgjqU5la3l/view?usp=sharing) |
| Cascade FPN RCNN EMD with RM | 0.9015 | 0.4032 | 0.8227 | [cascade_rcnn_emd_refine.mge](https://drive.google.com/file/d/1Cx7iyJGszruOjBEoi4V6jh9cUtpUj38k/view?usp=sharing) |
| RetinaNet FPN Baseline | 0.8769 | 0.4826 | 0.7716 | [retinanet_baseline.mge](https://drive.google.com/file/d/1QepMqiWx_e6Wb-XU8IgdByHCxrL42nlh/view?usp=sharing) |
| RetinaNet Simple       | 0.8977 | 0.4764 | 0.7919 | [retinanet_simple.pth](https://drive.google.com/file/d/1IjG092Ya0qPw-cQRwtG-8GfmcGVNiENi/view?usp=sharing)
