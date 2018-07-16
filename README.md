# KarfNN
 Another Library for Deep Neural Networks

## Getting Started

Below I will show you how to use KarfNN on your own projects

### Prerequisites

both python 2 and python 3 are supported so feel free to use the one you are comfortable with.

next, install all the packages mentioned in requirements.txt file
```
pip install  -r requirements.txt
```

### What is included
currently I have implemented the following:
##### Layers
* Dense Layer (fully_connected)
* Dropout
```python
from KarfNN.layer import Dense, Dropout
```
##### Activation functions
* linear
```python
Dense(n_units=...,activation="linear")
```
* relu
```python
Dense(n_units=...,activation="relu")
```
* softmax
```python
Dense(n_units=...,activation="softmax")
```
* sigmoid
```python
Dense(n_units=...,activation="sigmoid")
```
* tanh
```python
Dense(n_units=...,activation="tanh")
```
##### Losses
* Squared error
```
model.run(...,loss="squared_error")
```
* Cross entropy
```
model.run(...,loss="cross_entropy")
```
##### Optimizers
* Adam
```
model.run(...,optimizer="adam")
```
* Momentum
```
model.run(...,optimizer="momentum")
```
### Basic Exemples

##### Load the model
```python
from KarfNN.models import Karf
model = Karf()
```
##### initialize with data
```python
model.init(Xtrain, ytrain)
```
##### Add a Dense Layer
```python
from KarfNN.layer import Dense

model.add(Dense(n_units=10,activation="relu"))
```
##### Add a Dropout Layer
```python
from KarfNN.layer import Dropout
model.add(Dropout(treshold=0.8))
```
##### start training
```python
model.run(n_epochs=1500,batch_size=40,learning_rate=1e-4,optimizer="adam",loss="cross_entropy")
```

##### predict on your testing set
```python
model.predict(Xtest)
```

You can check the full Iris exemple in exemples folder
```
./exemples
```

Hope you like it ;-)
