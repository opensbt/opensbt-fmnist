# OpenSBT MNIST - Fashion-MNIST


This repository integrates the MNIST and Fashion-MNIST case study into OpenSBT and is based on the implementation in [DeepJanus](https://github.com/testingautomated-usi/deepjanus/blob/master/DeepJanus-MNIST/FULL_INSTALL.md).

## Preliminaries

Follow the installation instruction of OpenSBT in the [main repository](https://github.com/opensbt/opensbt-core). Install MNIST related dependencies by following the instructions [here](https://github.com/testingautomated-usi/deepjanus/blob/master/DeepJanus-MNIST/FULL_INSTALL.md
).

## Problem Definition

The [mutation](/mnist/mnist_simulation.py) of digits is performed by selection of control points/vertices of the corresponding svg model of the digit and performing a displacment in x and y direction of that point. The index of the point and extent represent the search parameters of the search based testing problem.
Right now, two configs are supported

- 3D Problem. (diplacement_1, displacement_2, index_vertex). Displaces vertex with given index and the following vertex with provided displacement.
- 6D Problem. (diplacement_1, displacement_2, displacement_3, diplacement_4, index_vertex_1, index_vertex_2). Displaces vertex_1 and vertex_2 with provided displacement.

There are multiple [fitness functions](/mnist/fitness_mnist.py) implemented integrating static image properties (num turns, saturation, etc.).

## Run example for MNIST

Example OpenSBT experiments can be found in [default_experiments.py](default_experiments_mnist.py).
To run an experiment via console use syntax from OpenSBT:

```bash
python run.py -e 1000
```

To see an example experiment definition without flag based execution check out ```run_mnist.py```.


## Run example for Fashion-MNIST

Example OpenSBT experiments can be found in [default_experiments_fmnist.py](default_experiments_fmnist.py).
To run an experiment via console use syntax from OpenSBT:

```bash
python run.py -e 201
```

To see an example experiment definition without flag based execution check out ```run_fmnist.py```.

## Authors

Lev Sorokin (lev.sorokin@tum.de)