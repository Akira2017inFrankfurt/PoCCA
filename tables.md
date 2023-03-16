Table 1:

|Category | Method |ModelNet40 ACC |ScanObjectNN ACC|
|:-|:-|:-:|:-:|
| Self-Supervised Generative-based|3D-GAN|83.3|-| 
| | Latent-GAN|85.7|-|  
| | SO-Net|87.3|-| 
| | FoldingNet|88.4|-| 
| | MRTNet|86.4|-| 
| | 3D-PointCapsNet| 88.9 |-|  
| | VIP-GAN|90.2|-|
| Self-Supervised Pretext tasks/Contrastive-based | PointNet + Jigsaw | 87.3 | 55.2 |
| |PointNet + STRL|88.3|*74.2*|
| |PointNet + Rotation|88.6|-|
| |PointNet + OcCo|88.7|69.5|
| |PointNet + CrossPoint|89.1|**75.6**|
| |PointNet + SelfCorrection|**89.9**|-| 
| |PointNet + PoCCA (Ours)|*89.4*|**75.6**|
| |DGCNN + ClusterNet|86.8|-|
| |DGCNN + Multi-Task|89.1|-|
| |DGCNN + Self-Contrast|89.6|-|
| |DGCNN + HSN|89.6|-|
| |DGCNN + Jigsaw|90.6|59.5|
| |DGCNN + STRL|90.9|77.9|
| |DGCNN + Rotation|90.8|-|
| |DGCNN + OcCo|89.2|78.3|
| |DGCNN + CrossPoint|*91.2*|*81.7*|
| |DGCNN + PoCCA (Ours)|**91.7**|**82.2**|


Table 2: 
|Category|Method|ModelNet40 ACC|ScanObjectNN ACC|
|:-|:-|:-:|:-:|
|Supervised|PointNet|89.2|68.2|
||PointNet++|90.7|77.9|
||PointCNN|92.2|78.5|
||DGCNN|92.9|78.1|
||PCT|93.2|-|
|Self-supervised(Encoder fine-tuned)|PointNet + Jigsaw|89.6|76.5|
||PointNet + Info3D|90.2|-|
||PointNet + OcCo|90.1|*80.0*|
||PointNet + SelfCorrection|90.0|-|
||PointNet + ParAEParAE|**90.5**|-|
||PointNet + PoCCA (Ours)|*90.2*|**80.3**|
||DGCNN + Jigsaw|92.4|82.7|
||DGCNN + OcCo|93.0|*83.9*|
||DGCNN + ParAE|92.9|-|
||DGCNN + Info3D|93.0|-|
||DGCNN + STRL|*93.1*|-|
||DGCNN + PoCCA (ours)|**93.3**|**84.1**|
