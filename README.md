# PoCCA
3D Point Cloud Understanding, Contrastive Learning

---------------------------------------------------------------
Environments:

- Python 3.7
- Cuda 10.2
- torch 1.8.1
- torchvision 0.9.1
- timm 0.6.12

---------------------------------------------------------------
Train steps:

- Step 1: run trainer.py to pretrain the model with ShapeNet dataset;
- Step 2: run cls_eval_svm.py to get linear classification results;
- Step 3: run cls_eval_fsl.py to get Few-Shot-Learning results;
- Step 4: run cls_finetune.py to get finetune classification results;
- Step 5: run tSNE_plot to get t-SNE figure;

---------------------------------------------------------------
Dataset:
- shapenet dataset
- modelnet40 dataset
- scanobjectnn dataset
- shapenet_part dataset

Events file:
- pretrain events
- local finetune events

Weights file:
- pretrained model weights
- finetune model weights
---------------------------------------------------------------


