# tinygrad

A simple autodiff engine with a basic neural network framework written to learn the inner workings of autograd! The project is meant to be syntactically similar to PyTorch (with <1% the functionality; it's for learning purposes only!)

It supports **scalar** values only! That is, it doesn't support matrices.

## What's supported?

**Automatic backward differentiation**

Yes

**Arithmetic Functions**

- Addition/Substraction/Multiplication/Division

- sin

- cos

- tan

- cot

- tanh

- pow

- log (any base)

- exp

- abs

**Non Linear Activation Functions**

- ReLU

**Layers**

- Linear Layers (only scalar values supported)

**Loss Functions**

- MSELoss

- MaxMarginLoss

- L1Loss

- SmoothL1Loss

**Optimizers**

Tried to conform to PyTorch's implementation of the following optimizers.

- SimpleSGD

- RMSProp

- Adam

- AdaGrad

- AdaMax

- AdaDelta

- AdamW

**Model Saving and Loading**

Rudimentary saving and loading but Yes! Saves and loads gradients also but can be made to save and load only weights and not the gradients.

## Demo

Check [demo1.ipynb](https://github.com/pranjaldatta/tinygrad/blob/master/demo1.ipynb) and [demo2.ipynb](https://github.com/pranjaldatta/tinygrad/blob/master/demo2.ipynb) for comparison with [micrograd](https://github.com/karpathy/micrograd). Check [demo](https://github.com/pranjaldatta/tinygrad/tree/master/demo) for more detailed demos on common datasets.

## Usage 

- Clone the repo!. Paste the following code into your terminal

```

$ git clone https://github.com/pranjaldatta/tinygrad.git
```

- cd into the root repo

```

$ cd tinygrad
```

- pip install it

```

$ pip install .
```

## Whats Next?

Redo the entire repo to support vectors!

## Credits

Inspired by Andrej Karpathy's repo [here](https://github.com/karpathy/micrograd).