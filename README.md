Implementation of NIPS2023: 
Unleashing the Full Potential of Product Quantization for Large-Scale Image Retrieval

The current code is the implementation on Glint360k. The training pipeline has been shown in run.sh, and will get the similar results(depends on different GPUs, drivers, etc).
Moreover,  We will upload the pretrained weights and the PQ-code label soon.

&emsp;[cisip-FIRe project](https://github.com/CISiPLab/cisip-FIRe)

&emsp;[insightface project](https://github.com/deepinsight/insightface)


## Datasets

### Large Scale Dataset

&emsp;The datasets download dir ref from [here](https://github.com/deepinsight/insightface/tree/c2db41402c627cab8ea32d55da591940f2258276/recognition/_datasets_)

&emsp;① Training dataset: Glint360K (360K ids/17M images)

&emsp;&emsp;[baidu](https://pan.baidu.com/s/1GsYqTTt7_Dn8BfxxsLFN0w) (code:o3az)

&emsp;② Testing dataset: Megaface + Facescrub 

&emsp;&emsp; [GDrive](https://drive.google.com/file/d/1KBwp0U9oZgZj7SYDXRxUnnH7Lwvd9XMy/view?usp=sharing)

&emsp;&emsp; Other file([noise sample remove](https://drive.google.com/drive/folders/14GXWU0f3SB4Bt4dF_jjMsB2OXzi4q4zv?usp=sharing))

&emsp;**Directory Structure**
```
./dataset/
  glint360k/
    train.idx
    train.rec
```
### Smaller Scale Dataset

&emsp;[ImageNet1K](https://image-net.org/download-images)

&emsp;[ImageNet100](https://drive.google.com/file/d/0B7IzDz-4yH_HSmpjSTlFeUlSS00/view?usp=drive_link&resourcekey=0-ozGVTlPhCjlY351mdV_9hg)

&emsp;**Directory Structure**
```
./dataset/
  imagenet1k/
    train/
        ...
    val/
        ...
    query/         #get by split val/
        ...
    gallery/       #get by split val/
        ...
```

## Pretrained Model
&emsp;Pretrained iResnet18、iResnet50、iResnet100 in Glint360K: [here-uploading]

&emsp;Pretrained Resnet50 in Imagenet1K: [here](https://download.pytorch.org/models/resnet50-19c8e357.pth)
