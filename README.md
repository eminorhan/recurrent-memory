# Recurrent Memory

This repository contains the code for reproducing the results reported in the following paper:

Orhan AE, Ma WJ (2019) [A diverse range of factors affect the nature of neural representations underlying short-term memory.](https://www.biorxiv.org/content/early/2018/10/13/244707) *Nature Neuroscience*.

The code is written in [Theano](http://www.deeplearning.net/software/theano/) (0.8.2) + [Lasagne](http://lasagne.readthedocs.io/en/latest/) (0.2.dev1). The code was originally run on a local computer cluster. If you are interested in running the following experiments on a cluster, I have some simple shell scripts that can facilitate this. Please contact me about this or about any other questions or concerns. You can find my contact information on [my web page](https://sites.google.com/view/eminorhan).

## Experiments

As described in the paper, there are six main experimental conditions.

* To run experiments in the basic condition:
```
python run_basic_expts.py --task 0 --model 0 --lambda_val 0.98 --sigma_val 0.0 --rho_val 0.0
```
where `task` is the integer code for the task, `model` is the integer code for the model, `lambda_val` is the value of the lambda_0 hyper-parameter, `sigma_val` is the value of the sigma_0 hyper-parameter, and `rho_val` is the value of the rho hyper-parameter in the paper. For the tasks reported in the paper, use the following integer codes for `task`: DE-1 (0), DE-2 (1), CD (2), GDE (4), 2AFC (6), Sine (7), COMP (8).   

* To run experiments in the Hebbian synaptic plasticity condition:
```
python run_basic_expts.py --task 0 --model 1 --lambda_val 0.98 --sigma_val 0.0 --rho_val 0.0
```
i.e. set the `model` argument to 1. This uses the model with fast weights as described in the paper.

* To run experiments in the tethered condition:
```
python run_tethered_expts.py --task 0 --model 0 --lambda_val 0.98 --sigma_val 0.0 --rho_val 0.0
```

* To run experiments in the dynamic input condition:
```
python run_dynamic_expts.py --task 0 --model 0 --lambda_val 0.98 --sigma_val 0.0 --rho_val 0.0
```
* To run experiments in the variable delay duration condition:
```
python run_vardelay_expts.py --lambda_val 0.98 --sigma_val 0.0 --rho_val 0.0
```
* The directory `multitask` contains files pertaining to the multitask training condition.

## Analysis

`utils.py` contains the function `compute_SI` that demonstrates how to compute the mean sequentiality index (SI) for a batch of trials as described in the paper.

