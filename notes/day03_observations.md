Answering the question: Is the CNN better because of structure, or because of fewer parameters?

CNN params: 1070794
MLP params: 1841162

Next step is to answer the question by reducing params of MLP to near CNN.
Hidden layers of MLP are changed to 315, which leads new params to be

MLP params: 1070695
CNN params: 1070794

with a difference of 99 (negligible).

From the new graphs, reducing MLP params to match CNN mostly reduces the training fit, not validation performance. This means that the MLP was already bottlenecked by representation, not capacity.

Answering another question: Which part of the CNN's inductive bias matters most?

To test this, this will be done through 3 tests
1. Remove downsampling
2. Remove locality
3. Remove translation equivariance/weight sharing

First one to be tested will be [2].
From testing,

1x1 convolution still has weight sharing, translation equivariance, depth/hierarchy, BUT

cannot aggregate spatial neighborhoods (each pixel location is processed independently accross channels.)