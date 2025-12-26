# Fixed Task
- Task: Image classification
- Dataset: CIFAR-10
- Train/val/test split: CIFAR-10 Standard Split (on website)

# Constants
- Optimizer: SGD + Momentum (target is to study inductive bias, using AdamW can cause blurs)
- LR Schedule: Cosine decay (smoother than Step decay, usually less fiddly)
- Batch Size: 128 (common across many models; LR Scaling Rule | LR = 0.1 * (batch_size / 128))
- Epochs: 100
- Augmentations: 
    - Train:
        - RandomCrop(32, padding=4)
        - RandomHorizontalFlip(p=0.5)
        - Normalize(CIFAR-10 mean/std)
    - Test/Val:
        - Normalize only
- Training seed(s): [0, 1, 2], report mean +- std
- Parameter Budget Rule: Match parameter count within +-10% across models
    - adjust width (channels/hidden units) until params match

- Momentum: 0.9
- Nesterov: True
- Weight Decay: 5e-4
- Loss: Cross-Entropy
- Gradient Clipping: Off (unless hit instability)

# Metrics Logging
- Train loss, train accuracy
- Val/test accuracy
- Learning curves across epochs
- Model parameter count
- Confidence histogram or entropy

# Analysis Files
- JSON/CSV row per run
- Confusion matrix on test set
- Small batch of failure cases: top-10 confident wrong predictions