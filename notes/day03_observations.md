Answering the question: Is the CNN better because of structure, or because of fewer parameters?

CNN params: 1070794
MLP params: 1841162

Next step is to answer the question by reducing params of MLP to near CNN.
Hidden layers of MLP are changed to 315, which leads new params to be

MLP params: 1070695
CNN params: 1070794

with a difference of 99 (negligible).

From the new graphs, reducing MLP params to match CNN mostly reduces the training fit, not validation performance. This means that the MLP was already bottlenecked by representation, not capacity.