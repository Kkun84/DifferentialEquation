# Differential Equation

This repository solves differential equations using neural networks.
Approximates the expression that is the solution to a differential equation with a neural network.
To learn the underlying equations of a system, it is enough to have the differential equations that the system follows and some real data.

It was not designed with utility in mind, but rather as an experiment.
For example, the point at which it takes a solution to learn a solution.

## Training Model

To train the model, enter the following command.

```
python src/train.py
```

The results of the training are output to `trained/`.
The learning process is recorded in TensorBoard.
TensorBoard is launched by the following command.

```
tensorboard --logdir=trained
```

## Change Target Differential Equation

The differential equations of the target and the equations of their solutions are written in `equation.py`.
The differential equation is written in the `differential_equation` function, and the equation of the solution in the `expression` function.
If you do not know the solution to a differential equation, it is not supported.
(I know it sounds funny that it takes a solution to learn a solution, but...)

## File Description
****
```
├── README.md
├── requirements.txt
│
├── src                 Main src files.
│   ├── train.py        Training model.
│   ├── equation.py     Definitions of differential equations and expression.
│   ├── model.py        Definition of model architecture.
│   └── dataset.py      Definition of dataset.
│
└── trained/        Output files by train.py.
```
