# Kaggle-Competition-Jaguar-Re-identification

## Project Structure

```
├── data/                    # Dataset from Kaggle
├── src/                     # Shared source code
│   ├── training/            # Training pipelines and scripts
│   ├── models/              # Model architectures and wrappers
│   └── util/                # Utility and helper functions
├── checkpoints/             # Saved model checkpoints
├── notebooks/               # Jupyter notebooks for experiments and analysis
├── output/                  # Submission output CSV files
└── README.md                # Project documentation
```


## Environment Management

Our code needs to run both in local environments and on Kaggle. To support this, we created a `setup.py` file inside the `/notebooks` directory. This file can be imported, and initializing the `Env()` class will automatically load the required environment variables.

### Local Development

For local development, copy `.env.sample` to `.env` and populate it with the appropriate values for your environment.

### Kaggle

For Kaggle, `setup.py` must be added as a requirement input. It will clone our GitHub repository and update the project path accordingly. Please note that all variables defined in `.env.sample` must be configured as Kaggle secrets.

### Experiments
- Domain Understanding:
    - Image dataset analysis, Embedding analysis
- Hyperparameter Optimization
    - Megadescriptor + Arc Face Loss with sweeps
- Model Architecture
    - Foundation models comparision: MegaDescriptor, DINOv3, ConvNeXt v2, Swin Transformers
    - Compare loss functions: ArcFace, Triplet Loss, combined losses
- Data & Training
    - Data augmentation (geometric, color, Cutout, MixUp, CutMix) + Test-time augmentation (TTA)
    - Classical Approaches: HotSpotter algorithm



- maybe to more than this