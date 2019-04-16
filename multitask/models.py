# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:57:28 2017 by emin
"""
import numpy as np
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer, GRULayer
from LeInit import LeInit
from CustomRecurrentLayerWithFastWeights import CustomRecurrentLayerWithFastWeights
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init

def bounded_relu(x):
    return T.clip(T.nnet.relu(x),0.0,100.0)

def OrthoInitRecurrent(input_var, mask_var=None, batch_size=1, n_in=100, n_out=1, n_hid=200, init_val=0.9, out_nlin=lasagne.nonlinearities.linear):
    # Input Layer
    l_in         = InputLayer((batch_size, None, n_in), input_var=input_var)
    if mask_var==None:
        l_mask=None
    else:
        l_mask = InputLayer((batch_size, None), input_var=mask_var)

    _, seqlen, _ = l_in.input_var.shape
    
    l_in_hid     = DenseLayer(lasagne.layers.InputLayer((None, n_in)), n_hid,  W=lasagne.init.GlorotNormal(0.95), nonlinearity=lasagne.nonlinearities.linear)
    l_hid_hid    = DenseLayer(lasagne.layers.InputLayer((None, n_hid)), n_hid, W=lasagne.init.Orthogonal(gain=init_val), nonlinearity=lasagne.nonlinearities.linear)
    l_rec        = lasagne.layers.CustomRecurrentLayer(l_in, l_in_hid, l_hid_hid, nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask, grad_clipping=100)

    # Output Layer
    l_shp        = ReshapeLayer(l_rec, (-1, n_hid))
    l_dense      = DenseLayer(l_shp, num_units=n_out, W=lasagne.init.GlorotNormal(0.95), nonlinearity=out_nlin)
    
    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out        = ReshapeLayer(l_dense, (batch_size, seqlen, n_out))

    return l_out, l_rec

def LeInitRecurrent(input_var, mask_var=None, batch_size=1, n_in=100, n_out=1, n_hid=200, diag_val=0.9, offdiag_val=0.01, out_nlin=lasagne.nonlinearities.linear):
    # Input Layer
    l_in = InputLayer((batch_size, None, n_in), input_var=input_var)
    if mask_var==None:
        l_mask=None
    else:
        l_mask = InputLayer((batch_size, None), input_var=mask_var)

    _, seqlen, _ = l_in.input_var.shape
    
    l_in_hid = DenseLayer(lasagne.layers.InputLayer((None, n_in)), n_hid, W=lasagne.init.GlorotNormal(0.95), nonlinearity=lasagne.nonlinearities.linear)
    l_hid_hid = DenseLayer(lasagne.layers.InputLayer((None, n_hid)), n_hid, W=LeInit(diag_val=diag_val, offdiag_val=offdiag_val), nonlinearity=lasagne.nonlinearities.linear)
    l_rec = lasagne.layers.CustomRecurrentLayer(l_in, l_in_hid, l_hid_hid, nonlinearity=lasagne.nonlinearities.rectify, mask_input=l_mask, grad_clipping=100)

    # Output Layer
    l_shp = ReshapeLayer(l_rec, (-1, n_hid))
    l_dense = DenseLayer(l_shp, num_units=n_out, W=lasagne.init.GlorotNormal(0.95), nonlinearity=out_nlin)

    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out = ReshapeLayer(l_dense, (batch_size, seqlen, n_out))

    return l_out, l_rec


def TanhRecurrent(input_var, mask_var=None, batch_size=1, n_in=100, n_out=1,
                    n_hid=200, wscale=1.0,
                    out_nlin=lasagne.nonlinearities.linear):
    # Input Layer
    l_in = InputLayer((batch_size, None, n_in), input_var=input_var)
    if mask_var == None:
        l_mask = None
    else:
        l_mask = InputLayer((batch_size, None), input_var=mask_var)

    _, seqlen, _ = l_in.input_var.shape

    l_in_hid = DenseLayer(lasagne.layers.InputLayer((None, n_in)), n_hid,
                          W=lasagne.init.HeNormal(0.95), nonlinearity=lasagne.nonlinearities.linear)
    l_hid_hid = DenseLayer(lasagne.layers.InputLayer((None, n_hid)), n_hid,
                           W=lasagne.init.HeNormal(gain=wscale), nonlinearity=lasagne.nonlinearities.linear)
    l_rec = lasagne.layers.CustomRecurrentLayer(l_in, l_in_hid, l_hid_hid, nonlinearity=lasagne.nonlinearities.tanh,
                                                mask_input=l_mask, grad_clipping=100)

    l_shp_1 =  ReshapeLayer(l_rec, (-1, n_hid))
    l_shp_2 =  ReshapeLayer(l_hid_hid, (-1, n_hid))

    l_shp = lasagne.layers.ElemwiseSumLayer((l_shp_1,l_shp_2),coeffs=(np.float32(0.2),np.float32(0.8)))

    # Output Layer
    l_dense = DenseLayer(l_shp, num_units=n_out, W=lasagne.init.HeNormal(0.95), nonlinearity=out_nlin)

    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out = ReshapeLayer(l_dense, (batch_size, seqlen, n_out))

    return l_out, l_rec


def LeInitRecurrentWithFastWeights(input_var, mask_var=None, batch_size=1, n_in=100, n_out=1,
                    n_hid=200, diag_val=0.9, offdiag_val=0.01,
                    out_nlin=lasagne.nonlinearities.linear, gamma=0.9):
    # Input Layer
    l_in = InputLayer((batch_size, None, n_in), input_var=input_var)
    if mask_var==None:
        l_mask=None
    else:
        l_mask = InputLayer((batch_size, None), input_var=mask_var)

    _, seqlen, _ = l_in.input_var.shape
    
    l_in_hid = DenseLayer(lasagne.layers.InputLayer((None, n_in)), n_hid,  
                          W=lasagne.init.GlorotNormal(0.95), 
                          nonlinearity=lasagne.nonlinearities.linear)
    l_hid_hid = DenseLayer(lasagne.layers.InputLayer((None, n_hid)), n_hid, 
                           W=LeInit(diag_val=diag_val, offdiag_val=offdiag_val), 
                           nonlinearity=lasagne.nonlinearities.linear)
    l_rec = CustomRecurrentLayerWithFastWeights(l_in, l_in_hid, l_hid_hid, 
                                                nonlinearity=lasagne.nonlinearities.rectify,
                                                mask_input=l_mask, grad_clipping=100, gamma=gamma)

    # Output Layer
    l_shp = ReshapeLayer(l_rec, (-1, n_hid))
    l_dense = DenseLayer(l_shp, num_units=n_out, W=lasagne.init.GlorotNormal(0.95), nonlinearity=out_nlin)
    
    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out = ReshapeLayer(l_dense, (batch_size, seqlen, n_out))

    return l_out, l_rec


def GRURecurrent(input_var, mask_var=None, batch_size=1, n_in=100, n_out=1, n_hid=200, diag_val=0.9, offdiag_val=0.01, out_nlin=lasagne.nonlinearities.linear):
    # Input Layer
    l_in         = InputLayer((batch_size, None, n_in), input_var=input_var)
    if mask_var==None:
        l_mask = None
    else:
        l_mask = InputLayer((batch_size, None), input_var=mask_var)
        
    _, seqlen, _ = l_in.input_var.shape
    l_rec        = GRULayer(l_in, n_hid, 
                            resetgate=lasagne.layers.Gate(W_in=lasagne.init.GlorotNormal(0.05), 
                                                          W_hid=lasagne.init.GlorotNormal(0.05), 
                                                          W_cell=None, b=lasagne.init.Constant(0.)), 
                            updategate=lasagne.layers.Gate(W_in=lasagne.init.GlorotNormal(0.05), 
                                                           W_hid=lasagne.init.GlorotNormal(0.05), 
                                                           W_cell=None), 
                            hidden_update=lasagne.layers.Gate(W_in=lasagne.init.GlorotNormal(0.05), 
                                                              W_hid=LeInit(diag_val=diag_val, offdiag_val=offdiag_val), 
                                                              W_cell=None, nonlinearity=lasagne.nonlinearities.rectify), 
                            hid_init = lasagne.init.Constant(0.), backwards=False, learn_init=False, 
                            gradient_steps=-1, grad_clipping=10., unroll_scan=False, precompute_input=True, mask_input=l_mask, only_return_final=False)

    # Output Layer
    l_shp        = ReshapeLayer(l_rec, (-1, n_hid))
    l_dense      = DenseLayer(l_shp, num_units=n_out, W=lasagne.init.GlorotNormal(0.05), nonlinearity=out_nlin)
    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out        = ReshapeLayer(l_dense, (batch_size, seqlen, n_out))

    return l_out, l_rec
