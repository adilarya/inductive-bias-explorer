Fixed Task
-> Task: Image classification
-> Dataset: CIFAR-10
-> Train/val/test split: CIFAR-10 Standard Split (on website)

Constants
-> Optimizer: SGD + Momentum (target is to study inductive bias, using AdamW can cause blurs)
-> LR Schedule: _
-> Batch Size: _
-> Epochs: 100
-> Augmentations: _
-> Training seed(s): _
-> Parameter Budget Rule: _


-> Momentum: 0.9
-> Nesterov: True
-> Weight Decay: 5e-4
-> Loss: Cross-Entropy
-> Gradient Clipping: Off (unless hit instability)

Metrics Logging
-> Train loss, train accuracy
-> Val/test accuracy
-> Learning curves across epochs
-> Model parameter count
-> Confidence histogram or entropy

Analysis Files
-> JSON/CSV row per run
-> Confusion matrix on test set
-> Small batch of failure cases: top-10 confident wrong predictions