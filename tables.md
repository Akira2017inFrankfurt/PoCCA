|Method|Backbone|Extra Training Data|ModelNet40 ACC|ScanObjectNN ACC|
|:-|:-:|:-:|:-:|:-:|
|SO-Net|SO-Net-Encoder|no|87.3|-|
|FoldingNet|FoldingNet-Encoder|no|88.4|-| 
|MRTNet|MRT-Encoder|no|86.4|-| 
|3D-PointCapsNet|3D Capsule-Encoder|no| 88.9 |-|  
|VIP-GAN|EncoderRNN|yes|90.2|-|
|IAE|DGCNN|yes|92.1|-|
|Point-M2AE|Hierarchical Transformer|yes|92.9|84.1|
|I2P-MAE|Hierarchical Transformer|yes|**93.4**|**87.1**|
|Latent-GAN|PointNet|no|85.7|-|
|Jigsaw|PointNet|no|87.3|55.2|
|STRL|PointNet|no|88.3|74.2|
|Rotation|PointNet|no|88.6|-|
|OcCo|PointNet|no|88.7|69.5|
|CrossPoint|PointNet|yes|89.1|**75.6**|
|SelfCorrection|PointNet|no|**89.9**|-| 
|PoCCA (Ours)|PointNet|no|*89.4*|**75.6**|
|ClusterNet|DGCNN|no|86.8|-|
|Multi-Task|DGCNN|no|89.1|-|
|Self-Contrast|DGCNN|no|89.6|-|
|HSN|DGCNN|no|89.6|-|
|Jigsaw|DGCNN|no|90.6|59.5|
|STRL|DGCNN|no|90.9|77.9|
|Rotation|DGCNN|no|90.8|-|
|OcCo|DGCNN|no|89.2|78.3|
|CrossPoint|DGCNN|yes|*91.2*|*81.7*|
|PoCCA(Ours)|DGCNN|no|**91.7**|**82.2**|


|Category|Method|Backbone|Extra Training Data|ModelNet40 ACC|ScanObjectNN ACC|
|:-|:-|:-:|:-:|:-:|:-:|
|Supervised|PointNet|-|no|89.2|68.2|
||PointNet++|-|no|90.7|77.9|
||PointCNN|-|no|92.2|78.5|
||DGCNN|-|no|92.9|78.1|
|Self-supervised(Encoder fine-tuned)|Jigsaw|PointNet|no|89.6|76.5|
||Info3D|PointNet|no|90.2|-|
||OcCo|PointNet|no|90.1|80.0|
||SelfCorrection|PointNet|no|90.0|-|
||ParAE|PointNet|no|90.5|-|
||PoCCA(Ours)|PointNet|no|90.2|80.3|
||Jigsaw|DGCNN|no|92.4|82.7|
||OcCo|DGCNN|no|93.0|83.9|
||ParAE|DGCNN|no|92.9|-|
||Info3D|DGCNN|no|93.0|-|
||STRL|DGCNN|no|93.1|-|
||PoCCA(ours)|DGCNN|no|93.3|84.1|
||ReCon|PointNet + Transformer|yes|93.0|83.8|
||PointBERT|PointNet + Transformers|no|93.2|83.1|
||PointMAE|PointNet + Transformer|no|93.8|85.2|
||MaskPoint|PointNet + Transformer|no|93.8|84.6|
||Point-M2AE|Hierarchical Transformer|yes|94.0|86.43|
||IAE|DGCNN|yes|94.2|-|

Table 6:

|Category|Method|Backbone|Extra Training Data|Class mIoU|Instance mIoU|
|:-|:-:|:-:|:-:|:-:|:-:|
|Supervised|PointNet|-|-|80.4|83.7|
||PointNet++|-|-|81.9|85.1|
||DGCNN|82.3|-|-|85.1|
||RSCNN|84.0|-|-|86.2|
||PCT|83.1|-|-|86.4|
|Self-supervised(Encoder frozen)|CloudContext|DGCNN|no|-|81.5|
||HNS|79.9|DGCNN|no|82.3|
||CMCV|74.7|DGCNN|no|80.8|
||PoCCA(Ours)|DGCNN|no|80.8|83.7|
|Self-supervised(Encoder fine-tuned)|Point-BERT|Hierarchical Transformer|no|84.1|85.6|
||Point-MAE|PointNet + Transformer|no|-|86.1|
||ReCon|PointNet + Transformer|yes|84.8|86.4|
||Point-M2AE|Hierarchical Transformer|yes|84.9|86.5|
||I2P-MAE|Hierarchical Transformer|yes|85.1|86.7|
||MaskPoint|PointNet + Transformer|no|84.6|86.0|
||Jigsaw |DGCNN|no|83.1|85.3|
||CMCV|DGCNN|no|79.1|83.7|
||OcCo|DGCNN|no|84.4|-|
||CrossPoint|DGCNN|yes|-|85.5|
||PoCCA(Ours)|DGCNN|no|84.5|86.0|
