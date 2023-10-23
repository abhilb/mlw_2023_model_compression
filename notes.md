# Model Compression

## A Survey on Model Compression and Acceleration for Pretrained Language models

## Model Compression for Deep Networks: A Survey
Summarizes SOTA for model compression: 
* Model Pruning
* Parameter Quantization
* Low rank decomposition
* Distillation


### Model Pruning
:bulb: Removes unimportant parameters

Structured Pruning
Unstructured Pruning

:plus: 
* Structured Pruning -> Hardware Acceleration
* Unstructured Pruning -> Smaller network

:minus:
* Structured Pruning -> Accuracy Reduction
* Unstructured Pruning -> Irregular architecture -> Difficult to acceleratre


### Parameter Quantization
:bulb: Convert data type to make calculations simpler

* Post training Quantization
* Quantization aware Training

:plus:
* Less Memory
* More speed up
* Lower energy consumption

:minus:
* Longer to train
* Inflexible

### Low-rank decomposition
:bulb: Decompose the original tensor into several low-rank tensors

:plus:
* 

:minus:

### Knowledge Distillation
:bulb: A large complex teacher network is used to guide smaller and simpler student network. 


## Model Compression via Distillation and Quantization

Introduces two methods

* ___Quantized Distillation___
  Leverages distillation during the training process by incorportating distillation loss w.r.t teacher network into training of a smaller student network whose weights are quantized to a limited set of levels. 
  
* ___Differentiable Quantization___
  Optimizes the location of quantization points through stochastic gradient descent to better fit the behavior of the teacher model. 

Existing hypothesis is that large models help transform local minima into saddle points or to discover robust solutions which do not depend on precise weight values. If we go ahead with the above hypothesis, then it means we can comporess the model should be achievable without impacting accuracy. 

Two approaches 

1. Training Quantized Neural Networks
2. Compress already trained networks.

## Misc

### Todo

-[ ] What is QAT (Quantization aware training)
-[ ] Examples of QAT
-[ ] How Distillation is done in DINOv2?
