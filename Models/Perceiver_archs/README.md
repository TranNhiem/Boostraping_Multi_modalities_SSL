## Sub-project : perceiverIO architecture design
### In this folder, we will main apply the [krasserm's repo](https://github.com/krasserm/perceiver-io) as our baseline.

---

### The PerceiverIO structure :
### The PerceiverIO structure can be mainly divided into 3 components, namely *data modality specific* InputAdapter, *task specific* OutputAdapter, PerceiverIO. Furthermore, the PerceiverIO can be further divided into 2 sub-components, namely PerceiverEncoder, PerceiverDecoder.
> The PerceiverIO is placed in the *perceiver_module.py* file and the basic Input/Output Adapter is placed in the *task_spec_adapter.py*.

### Finally, we further develope the perceiverIO artifact by combining the aforementioned components for the specific task. All of the perceiver artifact is placed in the *perceiver_artifact.py* (perceiverIO structure level model).

---

### Sometimes, we don't rapidly modify some detailed params of the perceiverIO artifacts, such as max_freq for fourier-position encoding.
### We also provide the baseline configuration placed in *artifact_config.py* for the perceiverIO.

--- 

### TODO list :
- [x] Visual-aid architecture design.
- [ ] Folder arrangement (only reveal perceiver_artifact.py interface)

---

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

---

#### The test folder is placed with the root dir, which contains the test functions which allow us not only confirm the correctness of each module but also **reveal the usage** of each module.

**HHRI SSL-group member : Josef-Huang 2022/08/15**