# Exploratory Data Analysis Experiments

This document summarizes exploratory data analysis experiments conducted to better understand the dataset and analyze the behavior of embedding models used for jaguar re-identification.

The goals of these experiments were:

- to identify structural properties of the dataset
- to detect potential issues such as class imbalance or duplicate images
- to analyze embedding representations
- to investigate whether models rely on background information

All experiments were conducted before or alongside model training to guide subsequent modeling decisions.

Implementation details and visualizations can be found in the corresponding [notebooks](https://github.com/CarloColumbo/Kaggle-Competition-Jaguar-Re-identification).

## Exploratory Data Analysis Notebook

In [00_exploratory_data_analysis](notebooks/00_exploratory_data_analysis.ipynb), we explored and familiarized ourselves with the dataset. We used [FiftyOne](https://docs.voxel51.com/) to visualize samples and analyze specific dataset characteristics.


We focused on the following questions
1. **Image Quality**  
   What are the resolution characteristics and foreground-background ratios of the images?

2. **Identity Distribution**  
   How balanced is the distribution of images across jaguar identities?

3. **Duplicate Images**  
   Does the dataset contain duplicate or near-duplicate images that could bias training?

4. **Embedding Structure**  
   Do image embeddings cluster according to jaguar identity?

We created a [Wandb Run](https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/9mbyr0ms) even that no model training was involved.

### Image Quality Analysis

In camera trap datasets, a large portion of the image may consist of background vegetation or terrain. If the foreground object occupies only a small region, models may struggle to learn discriminative features.

We analyzed:

- image width and height distributions
- proportion of masked pixels representing background

Initial inspection revealed that in many images the jaguar occupies only a relatively small portion of the frame, while the majority of pixels correspond to background.

This observation raises the concern that models may learn background-specific cues, such as vegetation or camera location, rather than focusing exclusively on the jaguar’s coat pattern.

Example images with high background ratios are shown below.

![Masked Images](output/00_exploratory-data-analysis/masked_images.png)

### Identity Distribution

Re-identification datasets often exhibit a **long-tailed distribution** of identities, where some individuals are represented by many images while others appear only rarely. This imbalance can bias training toward frequently observed identities.

We analyzed the number of images per identity and visualized the distribution:

![Identity Distribution](output/00_exploratory-data-analysis/identity_distribution.png)

The dataset shows a clear class imbalance:

- a small number of identities account for a large fraction of the images
- several identities appear only in a limited number of samples

Such imbalance can reduce model performance for underrepresented individuals.

This observation motivated later experiments with **class balancing strategies during training**, which are documented in [Leaderboard Experiments](LEADERBOARD_EXPERIMENTS.md).


### Embedding Analysis

We generated embeddings for all images using the **MegaDescriptor** model.

Using these embeddings, we performed:

- identification of representative images
- identification of unique images

**Representative Images** - Images located near the center of clusters

We can see that the representative images cover a variety of different individuals. However, some individuals are missing, while others appear multiple times. For example, Medrosa has three images that are considered highly representative. Some test images also appear to be representative of the dataset. These images mostly consist of high-quality samples where the jaguar is clearly visible and its coat patterns can be easily observed.

![Representative Images](output/00_exploratory-data-analysis/representive_images.png)

**Unique Images** - Images located far from other samples in embedding space.  

These may contain:

- unusual viewpoints
- occlusions
- partial visibility of the animal
- special postures

![Unique Images](output/00_exploratory-data-analysis/unique_image.png)

These images may be harder for models to correctly classify.


### Duplicate Analysis

Camera trap datasets frequently contain burst sequences or slightly shifted frames that are nearly identical. If such duplicates appear in both training and validation sets, evaluation results may be overly optimistic.

We performed near-duplicate detection using similarity search in the embedding space produced by the MegaDescriptor model. Images with very high embedding similarity were flagged as potential duplicates. 

The analysis identified **1,359 near-duplicate images**.

These duplicates typically correspond to:

- sequential frames captured by the same camera
- small positional changes of the animal within the frame

While duplicates may not always harm training, they reduce the effective diversity of the dataset.

Future work could investigate **duplicate removal or filtering strategies**.

Example duplicates are shown below.

![Duplicated Images](output/00_exploratory-data-analysis/duplicated_images.png)



## Background Interventions

From the image quality analysis, we observed that a large fraction of pixels correspond to background. This raises the question:

**Does the embedding model rely on background information to distinguish individuals?**

To investigate this, we performed a set of controlled background interventions within [01_background_intervention](notebooks/01_background_interventions.ipynb).

We tested the following background interventions:

1. **Baseline:** Unmodified images.
2. **Constant Background:** All masked pixels are filled with a constant value.
3. **Noise Background:** All masked pixels are filled with random values.
4. **Extended Foreground Edges:** Background is removed, and the empty space is filled by extending the foreground.
5. **Foreground Sampling for Background Filling:** Background pixels are filled by randomly sampling non-masked regions of the same image.
6. **Constant Image Background:** A single fixed image is used as the background for all images.
7. **Blurred Background:** A Gaussian blur is applied only to the background pixels.

### Results

Experiments were performed using multiple random seeds.

| Intervention           | Seed 36      | Seed 34      | Seed 78      | Seed 9       | Seed 12      | Mean mAP     | Std (mAP)    | Mean Epoch | Avg Position |
| ---------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ---------- | ------------ |
| Baseline               | **0.810537** | 0.770037     | **0.775120** | 0.822747     | 0.795316     | 0.794751     | 0.022603     | 80.40      | 2.20         |
| Constant Background    | 0.754350     | 0.798946     | 0.740401     | 0.793912     | 0.768489     | 0.771220     | 0.024018     | 84.40      | 4.00         |
| Noise Background       | 0.760136     | 0.775250     | 0.757010     | 0.799823     | 0.750507     | 0.768545     | 0.019577     | 90.60      | 4.40         |
| Extend Foreground      | 0.798990     | 0.765949     | 0.756068     | 0.812497     | 0.749464     | 0.776594     | 0.027231     | 68.60      | 3.80         |
| Sampling Foreground    | 0.776228     | 0.762772     | 0.753942     | 0.808110     | 0.767584     | 0.773727     | 0.021310     | 83.60      | 4.40         |
| Const Background Image | 0.761393     | 0.772697     | 0.766144     | 0.796175     | 0.771862     | 0.773654     | 0.013239     | 75.40      | 4.00         |
| Blur Background        | 0.793350     | **0.804884** | 0.770682     | **0.834982** | **0.813057** | **0.803391** | 0.023788     | 87.00      | **1.20**     |

### Interpretation

Our findings suggest that the background is an important feature for the embedding model. Completely removing it, for example by replacing it with a constant color, drastically reduced the identity-balanced mAP on the validation set. Most interventions that strongly disturbed the background (noise, foreground sampling, constant background image, or extending the foreground) were unable to reach the baseline mAP.

The final intervention, where the background was blurred, slightly improved the mAP. This indicates that the background still contains useful information for distinguishing jaguars, suggesting that the model does not rely exclusively on coat patterns and may struggle to fully generalize across environments. Based on these results, we decided to use the **blurred background** version of the dataset for all subsequent experiments.

The public leaderboard score of **0.7** suggests that the intervention may have worsened the model. However, this result was obtained from a single seed, while the validation mAP across multiple seeds indicates that using blurred backgrounds is still a reasonable choice.



### Wandb runs
- Seed 36: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/kl1e2tix?nw=nwuserkarlschuetz
- Seed 34: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/gby3xk1d?nw=nwuserkarlschuetz
- Seed 78: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/dhv2xmbo?nw=nwuserkarlschuetz
- Seed  9: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/hfwowpvm?nw=nwuserkarlschuetz
- Seed 12: https://wandb.ai/karl-schuetz-hasso-plattner-institut/jaguar-reid-karl-matti-schuetz/runs/kl1e2tix?nw=nwuserkarlschuetz


# Key Findings

The exploratory analysis revealed several important properties of the dataset:

1. **Large background regions** are present in many images.
2. **Identity distribution is highly imbalanced.**
3. **Near-duplicate images are common.**
4. **Background information influences embedding quality.**

These insights guided later modeling decisions, including preprocessing choices and training strategies.