CNN1x1 (which removes locality)
reduces accuracy by 10%+, which means that locality is one of the most important inductive biases in CNNs for image classification.

HOWEVER, the CNN accuracy is still better than the parameter-matched MLP, which suggests that there are other factors at play.

Next thing to be tested is a CNN without pooling [1].

cnn_no_pool params: 16799434

From the graphs of the CNNNoPool, removing pooling does not significantly hurt final performance. 