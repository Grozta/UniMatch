# UniMatch for Medical Image Segmentation for 3D

We provide the official PyTorch implementation of our UniMatch in the scenario of **semi-supervised medical image segmentation**:

> **[Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.09910)**</br>
> [Lihe Yang](https://liheyoung.github.io), [Lei Qi](http://palm.seu.edu.cn/qilei), [Litong Feng](https://scholar.google.com/citations?user=PnNAAasAAAAJ&hl=en), [Wayne Zhang](http://www.statfe.com), [Yinghuan Shi](https://cs.nju.edu.cn/shiyh/index.htm)</br>
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2023*


## Results

**You can refer to our [training logs](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/medical/training-logs) for convenient comparisons during reproducing.**


## Getting Started

### Installation

```bash
cd UniMatch
conda create -n unimatch python=3.10.4
conda activate unimatch
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```


### Dataset

- FLARE: [image,mask,unlabel](https://drive.google.com/drive/folders/1x0l-bxte46QFn5K_ZJzBxp8HsscF8v6t)

Please modify your dataset path in configuration files.

```
├── [Your FLARE Path]
    ├── imagesTr
    │   ├── FLARE22_Tr_0001_0000.nii.gz
    │   ├── FLARE22_Tr_0002_0000.nii.gz
    ├── labelsTr
    │   ├── FLARE22_Tr_0001.nii.gz
    │   ├── FLARE22_Tr_0002.nii.gz
    └── unlabeled
        ├── FLARE22_case_1001.nii.gz
        ├── FLARE22_case_1002.nii.gz
```


## Usage

### UniMatch
#### 前处理
1. run dataset_helper.py 生成数据描述文件，该文件中包含了素有样本的路径
```bash
python dataset/dataset_helper.py 

```

2. run dataset/dataset_preprocessing.py 运行前处理的过程，前处理包含了
   - 非零值裁剪、调整朝向、设定统一的spacing
   - 输入为图像整合成一个npy文件，和一个属性变换的记录文件pkl

```bash
python dataset/dataset_preprocessing.py
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/medical/scripts/train.sh).


### Supervised Baseline

Modify the ``method`` from ``'unimatch'`` to ``'supervised'`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/medical/scripts/train.sh), and double the ``batch_size`` in configuration file if you use the same number of GPUs as semi-supervised setting (no need to change ``lr``). 
#### 训练


## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{unimatch,
  title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  booktitle={CVPR},
  year={2023}
}
```


## Acknowledgement

The processed ACDC dataset is borrowed from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).
