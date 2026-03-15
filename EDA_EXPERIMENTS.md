# Exploratory Data Analysis Experiments

Experiments focused on understanding the dataset and model behavior, building analysis tooling, and answering diagnostic questions.

## Image Dataset Analysis
- **Identity Distribution**: Analyzed the number of images per individual jaguar to understand class imbalance.
- **Duplicates or Near-Duplicates**: Identified potential duplicate images in the dataset.
- **Image Quality**: Examined image dimensions, masked pixels, and other quality metrics.
- **Background Patterns**: Investigated the role of background features in the dataset.

## Embedding Analysis
- **Clustering and Nearest Neighbors**: Explored feature representations to study cohesion and separation of embeddings.
- **Cohesion and Separation**: Evaluated the quality of embeddings for distinguishing between jaguar identities.

## Triplet or Quadruplet Generation and Mining Analysis
- **Mining Strategies**: Analyzed the effectiveness of triplet and quadruplet mining strategies for training.

## Interpretability
- **Sanity and Faithfulness Tests**: Conducted interpretability experiments, including Layer-wise Relevance Propagation (LRP), to ensure model predictions are explainable and reliable.

## Background Intervention Analysis
- **Definition of Interventions**: Precisely defined background interventions, such as:
  - **Baseline**: Unmodified images.
  - **Constant Background**: Replacing masked pixels with a constant value.
  - **Noise Background**: Filling masked pixels with random values.
  - **Extended Foreground Edges**: Extending foreground regions to fill the background.
  - **Foreground Sampling**: Using non-masked regions to fill the background.
  - **Constant Image Background**: Applying a fixed image as the background.
  - **Blurred Background**: Applying Gaussian blur to background pixels.
- **Impact Analysis**: Evaluated the effect of these interventions on identity-balanced mAP.

## Class Balance
- **Interventions**:
  - **Weighted Sampling**: Adjusted sampling probabilities to address class imbalance.
  - **Augmented Samples**: Generated additional samples for underrepresented classes.
  - **Combined Strategies**: Combined weighted sampling with augmentation.
- **Evaluation**: Monitored identity-balanced mAP to assess the effectiveness of interventions.

## References
- [Exploratory Data Analysis Notebook](notebooks/00_exploratory_data_analysis.ipynb)
- [Background Interventions Notebook](notebooks/01_background_interventions.ipynb)
- [Class Balance Notebook](notebooks/07_class_balance.ipynb)
