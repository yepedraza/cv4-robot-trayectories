### Training Models

## Descriptions

### [Tensorflow model]()
The evaluation of a compendium of models was carried out, where the model with the most precision was chosen under pre-established parameters:
* So that the variation of fit **epochs** did not influence the choice of the model; the [*"EarlyStop"*](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) of tensorflow was used.
* The weights **initializer** used was [*"RandomUniform"*](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomUniform) with the default parameters
* The **loss function** chosen was [*"Categorical Crossentropy"*](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) and the evaluation **metric** was *"Accuracy"*; all due to the approach of the problem
* The **learning rate** of the used optimizers will be the default one
  
**Models generated**
1. 