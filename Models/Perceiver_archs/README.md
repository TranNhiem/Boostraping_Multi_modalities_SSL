## Sub-project : perceiverIO architecture design
### In this folder, we will main apply the [krasserm's repo](https://github.com/krasserm/perceiver-io) as our baseline.

### The test folder is placed with the root dir, which contains the test functions which allow us not only confirm the correctness of each module but also **reveal the usage** of each module.

### The following code apply the einops package, i'll give an brief demo of those functions mainly used in this folder.
```
import torch
from einops import rearrange, repeate

a = torch.rand(3, 8, 9)
rearrange(a, 'c (r p) w -> c r p w', p=2).shape  # [3, 4, 2, 9]
rearrange(a, 'c ... -> c (...)').shape  # [3, 72]
repeate(a, 'c r p q', q=4).shape  # [3, 8, 9, 4]
```
> ### yes! very powerful as the first glance.. 

**HHRI SSL-group member : Josef-Huang 2022/08/08**