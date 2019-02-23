# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import os
import sys
import theano
import theano.tensor as T
import numpy as np
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init
from utils import build_generators, build_model, build_loss, build_performance, compute_SI
import scipy.io as sio

os.chdir(os.path.dirname(sys.argv[0]))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
job_idx    = int(os.getenv('jobidx'))
np.random.seed(job_idx)

diagval_range = np.linspace(0.80,0.98,10)
offdiag_range = np.linspace(0.0,0.018,10)
DD,OO         = np.meshgrid(diagval_range, offdiag_range)
DD            = DD.flatten()
OO            = OO.flatten()

diag_val      = DD[job_idx-1]
offdiag_val   = OO[job_idx-1]
m_ind         = 0 # mm[job_idx-1]
t_ind         = 2 # tt[job_idx-1]

# Models and model-specific parameters
model_list = [{"model_id":'LeInitRecurrent',"diag_val":diag_val,"offdiag_val":offdiag_val},
              {"model_id":'LeInitRecurrentWithFastWeights',"diag_val":diag_val,"offdiag_val":offdiag_val,"gamma":0.0007},
              {"model_id":'OrthoInitRecurrent',"init_val":diag_val},
              {"model_id":'ResidualRecurrent',"leak_inp":1.0, "leak_hid":1.0},
              {"model_id":'GRURecurrent',"diag_val":diag_val,"offdiag_val":offdiag_val},
              {"model_id":'LeInitRecurrentWithLayerNorm', "diag_val": diag_val, "offdiag_val": offdiag_val},
              ]

# Tasks and task-specific parameters
task_list = [{"task_id":'DE1', "n_out":1, "n_loc":1, "out_nonlin":lasagne.nonlinearities.linear},
             {"task_id":'DE2', "n_out":2, "n_loc":2, "out_nonlin":lasagne.nonlinearities.linear},
             {"task_id":'CD1', "n_out":1, "n_loc":1, "out_nonlin":lasagne.nonlinearities.sigmoid},
             {"task_id":'CD2', "n_out":1, "n_loc":2, "out_nonlin":lasagne.nonlinearities.sigmoid},
             {"task_id":'GDE2',"n_out":1, "n_loc":2, "out_nonlin":lasagne.nonlinearities.linear},
             {"task_id":'VDE1',"n_out":1, "n_loc":1, "max_delay":100, "out_nonlin":lasagne.nonlinearities.linear},
             {"task_id":'Harvey2012', "n_out":1, "sigtc":15.0, "stim_rate":1.0, "n_loc":1, "out_nonlin":lasagne.nonlinearities.sigmoid},
             {"task_id":'SINE', "n_out":1, "n_loc":1, "alpha":0.25, "out_nonlin":lasagne.nonlinearities.linear},
             {"task_id":'COMP', "n_out":1, "n_loc":1, "out_nonlin": lasagne.nonlinearities.sigmoid}
             ] 

# Task and model parameters
ExptDict = {"model": model_list[m_ind], 
            "task": task_list[t_ind],   
            "tr_cond": 'all_gains',
            "test_cond": 'all_gains',
            "n_hid": 500, 
            "n_in": 50,
            "batch_size": 50,
            "stim_dur": 25,
            "delay_dur": 100,
            "resp_dur": 25,
            "kappa": 2.0,
            "spon_rate": 0.1,
            "tr_max_iter": 25001,
            "test_max_iter": 2501}

# Build task generators
generator_first, test_generator_first = build_generators(ExptDict)

# Define the input and expected output variable
input_var, target_var = T.tensor3s('input', 'target')   

# Build the model 
l_out, l_rec = build_model(input_var, ExptDict)

# The generated output variable and the loss function
if ExptDict["task"]["task_id"] in ['DE1','DE2','GDE2','VDE1','SINE']:
    pred_var = lasagne.layers.get_output(l_out)
elif ExptDict["task"]["task_id"] in ['CD1','CD2','Harvey2012','Harvey2012Dynamic','Harvey2016','COMP']:
    pred_var = T.clip(lasagne.layers.get_output(l_out), 1e-6, 1.0 - 1e-6)

# Build loss
rec_act    = lasagne.layers.get_output(l_rec)
l2_penalty = T.mean(lasagne.objectives.squared_error(rec_act[:,-5:,:], 0.0)) * 1e-4
loss       = build_loss(pred_var, target_var, ExptDict) + l2_penalty

# Create the update expressions
params       = lasagne.layers.get_all_params(l_out, trainable=True)
updates      = lasagne.updates.adam(loss, params, learning_rate=0.0005)

# Compile the function for a training step, as well as the prediction function 
# and a utility function to get the inner details of the RNN
train_fn     = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
pred_fn      = theano.function([input_var], pred_var, allow_input_downcast=True)
rec_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_rec, get_details=True), allow_input_downcast=True)

# TRAINING COMP
s_vec, opt_vec, net_vec, infloss_vec_first = [], [], [], []
for i, (example_input, example_output, s, opt) in generator_first:
    score = train_fn(example_input, example_output)
    net   = pred_fn(example_input)
    s_vec.append(s)
    opt_vec.append(opt)
    net_vec.append(np.squeeze(net[:,-5,:]))
    if i % 500 == 0:
        opt_vec = np.asarray(opt_vec)
        net_vec = np.asarray(net_vec)
        s_vec   = np.asarray(s_vec)
        infloss = build_performance(s_vec,opt_vec,net_vec,ExptDict)
        infloss_vec_first.append(infloss)
        print '(First task) Batch #%d; X-ent: %.6f; Inf. loss: %.6f' % (i, score, infloss)
        s_vec   = []
        opt_vec = []
        net_vec = []

# TESTING COMP
s_vec, opt_vec, net_vec = [], [], []
for i, (example_input, example_output, s, opt) in test_generator_first:
    net = pred_fn(example_input)
    s_vec.append(s)
    opt_vec.append(opt)
    net_vec.append(np.squeeze( net[:,-5,:] ))     

opt_vec = np.asarray(opt_vec)
net_vec = np.asarray(net_vec)
s_vec   = np.asarray(s_vec)            
infloss_test_first = build_performance(s_vec,opt_vec,net_vec,ExptDict)
print '(First task) Test data; Inf. loss: %.6f' %infloss_test_first

# SET UP SECOND TASK
ExptDict["task"] = task_list[6] # 2AFC task
generator_second, test_generator_second = build_generators(ExptDict)

# TRAINING SECOND TASK
s_vec, opt_vec, net_vec, infloss_vec_second = [], [], [], []
for i, (example_input, example_output, s, opt) in generator_second:
    score = train_fn(example_input, example_output)
    net   = pred_fn(example_input)
    s_vec.append(s)
    opt_vec.append(opt)
    net_vec.append(np.squeeze(net[:,-5,:]))
    if i % 500 == 0:
        opt_vec = np.asarray(opt_vec)
        net_vec = np.asarray(net_vec)
        s_vec   = np.asarray(s_vec)
        infloss = build_performance(s_vec,opt_vec,net_vec,ExptDict)
        infloss_vec_second.append(infloss)
        print '(Second task) Batch #%d; X-ent: %.6f; Inf. loss: %.6f' % (i, score, infloss)
        s_vec   = []
        opt_vec = []
        net_vec = []

# TESTING SECOND TASK
s_vec, opt_vec, net_vec, ex_hid_vec = [], [], [], []
for i, (example_input, example_output, s, opt) in test_generator_second:
    net = pred_fn(example_input)
    example_hidden = rec_layer_fn(example_input)
    s_vec.append(s)
    opt_vec.append(opt)
    net_vec.append(np.squeeze( net[:,-5,:]))
    if i % 500 == 0:
        ex_hid_vec.append(example_hidden)

opt_vec = np.asarray(opt_vec)
net_vec = np.asarray(net_vec)
s_vec   = np.asarray(s_vec)
infloss_test_second = build_performance(s_vec, opt_vec, net_vec, ExptDict)
print '(Second task) Test data; Inf. loss: %.6f' %infloss_test_second

# Input and hidden layer activities
ex_hid_vec = np.asarray(ex_hid_vec)
ex_hid_vec = np.reshape(ex_hid_vec,(-1, test_generator_second.stim_dur + test_generator_second.delay_dur +
                                    test_generator_second.resp_dur, ExptDict["n_hid"]))

# COMPUTE SI
infloss_threshold = 0.5
entrpy_bins       = 20        # number of bins for computing entropy of peak resp. time distribution
window_size       = 5         # window around peak time to compute mean ridge activity
r_threshold       = 1e-1      # only consider neurons with mean activity above this

if (infloss_test_second < infloss_threshold):
    SI = compute_SI(ex_hid_vec, entrpy_bins, window_size, r_threshold)
else:
    SI = 0.0

print 'Mean SI %f'%SI

# SAVE TRAINED MODEL      
sio.savemat('500jobidx%i_model%i_cd1_har12_everything.mat'%(job_idx,m_ind),
            {'all_params_list':lasagne.layers.get_all_param_values(l_out, trainable=True),
             #'inpResps':ex_inp_vec,
             #'hidResps':ex_hid_vec,
             'SI':SI,
             'infloss_test_first':infloss_test_first,
             'infloss_vec_first':np.asarray(infloss_vec_first),
             'infloss_test_second': infloss_test_second,
             'infloss_vec_second': np.asarray(infloss_vec_second)
             })
