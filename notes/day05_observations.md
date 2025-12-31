Now the last test is removing the translation equivariance/weight sharing. 
To accomplish this, 
- local patches will be used
- but has different weights at each (i, j) location.

This removes weight sharing and translation equivariance while keeping locality. 

The size of the shape test is [2, 10]. 
cnn_lc1 params: 1987402

it is observed there is little reduction from removing the translation equivariance or weight sharing. The biggest impact comes from locality!