# Leaderboard Experiments

This document summarizes leaderboard experiments conducted to find a model architecture and training pipeline to maximize the performance on jaguar re-identification.

In each conducted experiment, we focus on increasing the identity-balanced mAP on the validation set. We try to keep the pipeline and model fixed and only change on part of this per experiment.

We focused on the following experiments:

1. Finding the optimal **backbone**:
    Which backbone has the best compromiss of predictive power for the jaguar ReId and efficiency.
2. Finding the optimal **loss function**:
    Which loss function can we use to get the best results for identity-balanced mAP. What characteristic do we need for the loss function?
3. **Combination of loss functions**:
    Is a combination of two different loss function better for model training than using one loss alone? What happens with the predictive power and efficiency?
4. Finding the best **optimizer**:
    Which optimizer enables smoothing and stable training?
5. Finding the best **scheduler**:
    Which scheduler can increase the training speed while preserving stable gradients with less noise?
6. Impact of **Class Balancing**:
    Can class balancing improve the predictive power and which intervention should be used?
7. Model **hyperparamter sweep**:
    What are good hyperparameter for our model desing and how influenced are they by seeding?
8. **Reranking** of final results:
    Can we improve the identity-balanced mAP through reranking of the final calculated cosine distances?
    

The experiments were conducted in this order. After each experiment we decided which component should be used in subsequent experiments. With this approach, we could improve our model incrementally. One problem is that wrong decision early one could worsend our final results.

Implementation details and visualizations can be found in the corresponding [notebooks](https://github.com/CarloColumbo/Kaggle-Competition-Jaguar-Re-identification).


## 1. Backbone Comparison

In this project we soly focused on the training of embedding projection models. For them, the most important thing is the backbone. The backbone provides initial embeddings which could be refinded by the projection model. The backbone needs to provide very descriptive embeddings for effective ReId.

We compared the following models in [02_backbone](notebooks/02_backbone.ipynb) (reasons for each is within the notebook):
1. **MegaDescriptor**
2. **CLIP**
3. **DINOv3**
4. **EfficientNet**
5. **ResNet18**

Backones were used to embedd all images. Afterwards, for each backbone we trained one embedding projection model on the embeddings while keeping the training setup except of the model input size the same.

### Results
| Backbone       | Seed 4       | Seed 7       | Seed 90      | Seed 856     | Seed 21      | Mean mAP     | Std (mAP) | Mean Epoch | Mean Embedding Time | Mean Training Time | Avg Position |
| -------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | --------- | ---------- | ------------------- | ------------------ | ------------ |
| MegaDescriptor | 0.845019     | 0.814579     | 0.784838     | 0.765074     | 0.733252     | 0.788552     | 0.043269  | 90.0       | 135.0               | 172.467            | 2.2          |
| CLIP           | 0.802713     | 0.794566     | 0.776417     | 0.735205     | 0.690422     | 0.759465     | 0.046756  | 130.6      | 90.8                | 234.356            | 3.6          |
| Dino           | **0.870783** | **0.845053** | **0.829684** | **0.812577** | **0.780221** | **0.827664** | 0.034074  | 83.4       | 104.7               | 174.314            | **1.0**      |
| EfficientNet   | 0.830311     | 0.762886     | 0.773556     | 0.780844     | 0.720779     | 0.773675     | 0.039286  | 90.6       | 141.1               | 196.480            | 3.4          |
| ResNet18       | 0.650435     | 0.664405     | 0.663641     | 0.626006     | 0.598869     | 0.640671     | 0.028057  | 90.2       | 136.4               | 187.938            | 5.0          |

The **DINOv3** model outperformed every other model by a huge step. It results in a very high mAP compared to other backbones, a lower variance in the mAP, and a faster trainings time. Using **DINOv3** also improved our public score significantly to 0.751.

For future experiments, we use **DINOv3** as backbone.

### Wandb runs:
- Seed   4: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/jb256kh5?nw=nwuserkarlschuetz
- Seed   7: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/xd11wr8o?nw=nwuserkarlschuetz
- Seed  90: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/m9xn33hd?nw=nwuserkarlschuetz
- Seed 856: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/mr3mohwf?nw=nwuserkarlschuetz
- Seed  21: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/ppjzs2xy?nw=nwuserkarlschuetz


## 2. Loss Comparision

We found a good backbone for our use case. Now, we need a good loss function for the training of the embedding projection model in [03_loss](notebooks/03_loss.ipynb). The loss function could highly increase the mAP and stabilize training. Also the best performing loss function can show what is more important for the latent space. For example some loss functions like **ArcFace Loss** increase the angular distance between classes/identities while loss functions like **Center Loss** reduce the intra-class distance.

The following loss functions are evaluated:

1. **ArcFace Loss**
2. **CosFace Loss**
3. **SphereFace Loss**
4. **Proxy Anchor Loss**
5. **Sub-Center ArcFace Loss**
6. **Center Loss**
7. **Batch-Hard Triplet Loss**

### Results

| Loss Function        | Seed 42      | Seed 66      | Seed 102     | Seed 305     | Seed 12      | Mean mAP     | Std (mAP)    | Mean Epoch | Mean Time  | Avg Position |
| -------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ---------- | ---------- | ------------ |
| ArcFaceLoss          | 0.810142     | 0.809076     | 0.793583     | 0.813424     | 0.845993     | 0.814444     | 0.019228     | 77.60      | 101.80     | 3.80         |
| CosFaceLoss          | 0.807405     | 0.813197     | 0.788920     | 0.800354     | 0.838721     | 0.809719     | 0.018567     | **73.40**  | **100.54** | 4.20         |
| SphereFaceLoss       | 0.831131     | **0.860190** | 0.812642     | 0.752520     | 0.863378     | 0.823972     | 0.045138     | 127.00     | 174.73     | 3.20         |
| ProxyAnchorLoss      | 0.829121     | 0.858875     | **0.839110** | 0.816946     | 0.854525     | 0.839715     | 0.017446     | 128.60     | 178.72     | 2.40         |
| SubCenterArcFaceLoss | 0.667916     | 0.700064     | 0.721361     | 0.699483     | 0.707221     | 0.699209     | 0.019589     | 105.60     | 159.25     | 7.00         |
| Center Loss          | **0.842462** | 0.849450     | 0.828360     | **0.847258** | **0.867817** | **0.847069** | **0.014208** | 165.40     | 231.33     | **1.40**     |
| Batch Hard Triplet   | 0.765055     | 0.732546     | 0.769427     | 0.758815     | 0.780067     | 0.761182     | 0.017785     | 101.00     | 160.86     | 6.00         |

**Center Loss** shows the best performance with the lowest variance. It seems that **Center Loss** is a promising loss function for our use case, which implies that the learned class centers are a good way to train the model because through them we reduce intra-class variance within the embedding space.

This loss could increase the public score to 0.786.

### Wandb runs:
- Seed 42: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/4emyw0ox?nw=nwuserkarlschuetz
- Seed 66: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/7p827dmg?nw=nwuserkarlschuetz
- Seed 102: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/uwbc18ao?nw=nwuserkarlschuetz
- Seed 305: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/e4va5g2w?nw=nwuserkarlschuetz
- Seed 12: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/30txt1y1?nw=nwuserkarlschuetz


## 3. Loss Combinations

In the previous experiment **Center Loss** as well as **Proxy Anchor Loss** seemed very promising for jaguar ReId. One possibility is to combine both loss functions into a new loss function. This procedure can combine the advantages of both. In the case of **Center Loss** and **Proxy Anchor Loss** we can combine reduced intra-class variance through **Center Loss** and reduced inter-class variance through **Proxy Anchor Loss**. We tried this approach in [04_loss_combined](notebooks/04_loss_combined.ipynb).

### Results
| Loss Function                   | Seed 3       | Seed 908     | Seed 45      | Seed 33      | Seed 123     | Mean mAP     | Std (mAP) | Mean Epoch | Mean Time   | Avg Position |
| ------------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | --------- | ---------- | ----------- | ------------ |
| Center Loss                     | 0.923984     | **0.928725** | 0.862973     | **0.880770** | 0.870704     | 0.893431     | 0.030756  | 159.6      | **220**     | 2            |
| Combined (Center + ProxyAnchor) | 0.926113     | 0.926888     | **0.867175** | 0.877847     | **0.891981** | **0.898001** | **0.027465**  | 169.8      | 268         | **1.8**  |
| Combined (Center + ArcFace)     | **0.930783** | 0.925359     | 0.843008     | 0.879712     | 0.880382     | 0.891849     | 0.036411  | 173.4      | 300         | 2.2          |

The combined loss of **Center Loss** and **Proxy Anchor Loss** outperformed the other loss combination and the current baseline using **Center Loss**. Therefore, **Center Loss** will be used in subsequent experiments.

The resulting public score was 0.806.

In the MDS plot below, one can see how tight the clusters for identies became through center loss.

![MDS Center + Proxy Anchor]](output/04_loss_combined/center_proxyanchor_embeddings_mds_finetuned.png)

### Wandb runs
- Seed 3: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/md9bvsih?nw=nwuserkarlschuetz
- Seed 908: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/zo0usamg?nw=nwuserkarlschuetz
- Seed 45: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/8u5evx5v?nw=nwuserkarlschuetz
- Seed 33: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/insejndd?nw=nwuserkarlschuetz
- Seed 123: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/hxbq8syi?nw=nwuserkarlschuetz


## 4. Optimizer Comparison

After finding a good backbone and loss function, we tried to improve the training using different optimizers in [05_optimizers](notebooks/05_optimizers.ipynb). 

We tested the following optimizer under the same training pipeline:

1. **Adam**
2. **AdamW**
3. **NAdam**
4. **SGD + Momentum**
5. **RMSProp**

### Results
| Optimizer | Seed 12    | Seed 67      | Seed 99      | Seed 87      | Seed 334     | Mean mAP     | Std (mAP) | Mean Epoch | Mean Training Time | Avg Position |
| --------- | ---------- | --------     | ------------ | ------------ | ------------ | --------     | --------- | ---------- | ------------------ | ------------ |
| Adam      | 0.884543   | **0.850196** | 0.871263     | 0.907684     | 0.838015     | 0.870340     | 0.027604  | 41.2       | 72.729             | 2.0          |
| AdamW     | **0.9008** | 0.846063     | **0.879358** | 0.901981     | **0.838438** | **0.873328** | 0.029886  | 46.2       | 83.259             | 1.6          |
| NAdam     | 0.896485   | 0.841654     | 0.864915     | **0.907889** | 0.835509     | 0.869290     | 0.032224  | 40.6       | 67.955             | 2.5          |
| SGD       | 0.833591   | 0.777868     | 0.818324     | 0.855106     | 0.794424     | 0.815863     | 0.030674  | 16.2       | 37.668             | 4.0          |
| RMSprop   | 0.745348   | 0.663232     | 0.707518     | 0.718692     | 0.650438     | 0.697046     | 0.039456  | 9.4        | **30.569**         | 5.0          |

**AdamW** outperformed the other optimizers. Previously we already used **AdamW** which means that we will not change this component in the training pipeline.

Astonishing we could get a better public score with 0.819.

### Wandb runs
- Seed 12: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/e6q5frhl?nw=nwuserkarlschuetz
- Seed 67: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/0lav3zhj?nw=nwuserkarlschuetz
- Seed 99: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/1m3r15pp?nw=nwuserkarlschuetz
- Seed 87: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/9a363p43?nw=nwuserkarlschuetz
- Seed 334: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/80bbzm3p?nw=nwuserkarlschuetz


## 5. Scheduler Comparison

The last training component is the scheduler. We compared multiple ones in [06_scheduler](notebooks/06_scheduler.ipynb) and adjusted the trainings function to support multiple types of schedulers.

Tested schedulers are:
1. **StepLR**
2. **CosineAnnealingLR**
3. **ReduceLROnPlateau**
4. **ExponentialLR**
5. **OneCycleLR**

### Results
| Scheduler         | Seed 34      | Seed 46  | Seed 78      | Seed 98      | Seed 234     | Mean mAP     | Std (mAP)     | Mean Epoch | Mean Training Time | Avg Position |
| ----------------- | ------------ | -------- | ------------ | ------------ | ------------ | ------------ | ---------     | ---------- | ------------------ | ------------ |
| StepLR            | 0.840217     | 0.846933 | 0.821254     | 0.831274     | 0.863379     | 0.840611     | 0.015972      | 23.6       | **46.132**         | 4.6          |
| CosineAnnealingLR | **0.857931** | **0.875787** | 0.853690     | 0.859898     | 0.890453     | 0.867552     | 0.015294      | 20.4       | 44.992             | 1.8          |
| ReduceLROnPlateau | 0.838675     | 0.858497 | 0.851659     | 0.867299     | 0.888780     | 0.860982     | 0.018731      | 184.2      | 269.648            | 3.4          |
| ExponentialLR     | 0.847357     | 0.865327 | 0.819214     | 0.850021     | 0.873608     | 0.851105     | 0.020862      | 25.8       | 55.050             | 3.8          |
| OneCycleLR        | 0.852396     | 0.868097 | **0.875455** | **0.875737** | **0.894362** | **0.873209** | **0.015151**  | 69.6       | 131.160            | **1.4**      |

### Wandb runs
- Seed 34: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/unkyt26r?nw=nwuserkarlschuetz
- Seed 46: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/o5z6vfhu?nw=nwuserkarlschuetz
- Seed 78: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/r8tfbx2l?nw=nwuserkarlschuetz
- Seed 98: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/8tu4sty8?nw=nwuserkarlschuetz
- Seed 234: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/qswnnsjg?nw=nwuserkarlschuetz