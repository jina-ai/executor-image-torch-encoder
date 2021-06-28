# ✨ ImageTorchEncoder

**ImageTorchEncoder** encodes `Document` content from a ndarray, potentially B x (Height x Width x Channel) into a ndarray of B x D.  
Internally, **ImageTorchEncoder** wraps the models from [torchvision](https://pytorch.org/vision/stable/index.html).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites


The [dependencies](requirements.txt) for this executor can be installed using `pip install -r requirements.txt`.
The test suite has additional [requirements](tests/requirements.txt).

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ImageTorchEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://ImageTorchEncoder'
    with: 
      model_name: 'mobilenet_v2'
``` 
This does not support GPU at the moment.

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ImageTorchEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://ImageTorchEncoder'
```

### 📦️ Via Pypi

1. Install the `executor-image-torch-encode` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-image-torch-encoder.git
	```

1. Use `executor-image-torch-encode` in your code

	```python
	from jina import Flow
	from jinahub.image.encoder import ImageTorchEncoder 
	
	f = Flow().add(uses=ImageTorchEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-image-torch-encoder.git
	cd executor-image-torch-encoder
	docker build -t executor-image-torch-encoder .
	```

1. Use `executor-image-torch-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-image-torch-encoder:latest')
	```
	

## 🎉️ Example 


```python
import numpy as np

from jina import Flow, Document

f = Flow().add(uses='docker://executor-image-torch-encoder')

with f:
    resp = f.post(on='foo', inputs=Document(blob=np.ones((3, 224, 224), dtype=np.float32)), return_resutls=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `blob` of the shape `H x W x C`. You can inform the executor about the channel axis  
of your input images with the `channel_axis` parameter. 
Note that the `ImageTorchEncoder` does not resize or normalize the image before inference but this is required to 
get high performance from the models. We have implemented an [ImageNormalizer](https://github.com/jina-ai/executor-image-normalizer) which does the job.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (size depends on the model) with `dtype=nfloat32`.


## 🔍️ Reference
- [PyTorch TorchVision Transformers Preprocessing](https://sparrow.dev/torchvision-transforms/)