# OpenSBT MNIST - Fashion-MNIST

This repository integrates the MNIST and Fashion-MNIST case study into OpenSBT and is based on the implementation in [DeepJanus](https://github.com/testingautomated-usi/deepjanus/blob/master/DeepJanus-MNIST/FULL_INSTALL.md).

## Preliminaries

### Download Repository ###

```bash
git clone https://github.com/opensbt/opensbt-fmnist.git
cd opensbt-fmnist
```

### Installing Python 3.8 ###
Install Python 3.8
``` 
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.8
```

And check if it is correctly installed, by typing the following command:

``` 
$ python3.8
```

You should have a message that tells you are using python 3.8

### Installing pip ###
Use the following commands to install pip and upgrade it to the latest version:
``` 
apt install -y python3-pip
python3.8 -m pip install --upgrade pip
```

Once the installation is complete, verify the installation by checking the pip version:

``` 
python3.8 -m pip --version
```
### Creating a Python virtual environment ###

Install the `venv` module in the docker container:

``` 
apt install -y python3.8-venv
```

Create the python virtual environment:

```
python3.8 -m venv .venv
```

Activate the python virtual environment and updated `pip` again (venv comes with an old version of the tool):

```
. .venv/bin/activate
pip install --upgrade pip
```
### Installing OpenSBT Requirements ###

```
pip  install -r requirements.txt
```


### Installing Python Binding to the Potrace library ###
Instructions provided by https://github.com/flupke/pypotrace.

Install system dependencies in your environment (it is not needed to install them in the MNIST folder):

``` 
apt-get install build-essential python3,8-dev libagg-dev libpotrace-dev pkg-config 
```

Install pypotrace (commit `76c76be2458eb2b56fcbd3bec79b1b4077e35d9e`):

```
git clone https://github.com/flupke/pypotrace.git
cd pypotrace
git checkout 76c76be2458eb2b56fcbd3bec79b1b4077e35d9e
pip install numpy
pip install setuptools==65.5.0 wheel Cython==0.29.34
pip install .
cd ..
```
If the following command does not crash, pypotrace is correctly installed:

``` 
python
>>> import potrace
>>>
```

### Installing PyCairo and PyGObject ###
Instructions provided by https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started.

Open a terminal and execute 

```apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0```

And

```apt-get install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 librsvg2-dev```

### Installing MNIST requirements ###

```
cd mnist/
pip install -r requirements.txt
cd ...
```


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

Lev Sorokin (lev.sorokin@tum.de)
