# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import os 
import sys
import argparse
import theano
import theano.tensor as T
import numpy as np
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init
import generators, models
import scipy.io as sio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recurrent Memory Experiment (Variable Delay Condition -DE2)')
    parser.add_argument('--lambda_val', type=float, default=0.98, help='lambda (initialization for diagonal terms)')
    parser.add_argument('--sigma_val', type=float, default=0.0, help='sigma (initialization for off-diagonal terms)')
    args = parser.parse_args()

    diag_val = args.lambda_val
    offdiag_val = args.sigma_val

    model_list = ['LeInitRecurrent','ResidualRecurrent','GRURecurrent']

    # Task and model parameters
    model      = model_list[0]
    tr_cond    = 'all_gains'
    test_cond  = 'all_gains'
    n_hid      = 500         # number of hidden units

    generator  = generators.VarDelayedEstimationTask(max_iter=25001, batch_size=50,
                                                         n_loc=2, n_in=50, n_out=2,
                                                         stim_dur=25, max_delay=100,
                                                         resp_dur=25, kappa=2.0,
                                                         spon_rate=0.1, tr_cond=tr_cond)
    test_generator = generators.VarDelayedEstimationTask(max_iter=2501, batch_size=50,
                                                         n_loc=2, n_in=50, n_out=2,
                                                         stim_dur=25, max_delay=100,
                                                         resp_dur=25, kappa=2.0,
                                                         spon_rate=0.1, tr_cond=test_cond)

    # Define the input and expected output variable
    input_var, target_var = T.tensor3s('input', 'target')
    mask_var = T.bmatrix('mask')

    if model == 'LeInitRecurrent':
        l_out, l_rec = models.LeInitRecurrent(input_var, mask_var=mask_var,
                                              batch_size=generator.batch_size,
                                              n_in=generator.n_loc*generator.n_in,
                                              n_out=generator.n_out, n_hid=n_hid,
                                              diag_val=diag_val, offdiag_val=offdiag_val,
                                              out_nlin=lasagne.nonlinearities.linear)
    elif model == 'ResidualRecurrent':
        l_out, l_rec = models.ResidualRecurrent(input_var, mask_var=mask_var, batch_size=generator.batch_size, n_in=generator.n_loc*generator.n_in, n_out=generator.n_out, n_hid=n_hid, leak_inp=1.0, leak_hid=1.0)
    elif model == 'GRURecurrent':
        l_out, l_rec = models.GRURecurrent(input_var, mask_var=mask_var, batch_size=generator.batch_size, n_in=generator.n_loc*generator.n_in, n_out=generator.n_out, n_hid=n_hid)

    # The generated output variable and the loss function
    pred_var     = lasagne.layers.get_output(l_out)
    rec_act      = lasagne.layers.get_output(l_rec)
    l2_penalty   = T.mean(lasagne.objectives.squared_error(rec_act[:,-5:,:], 0.0)) * 1e-4
    loss         = T.mean(T.mod(T.abs_(pred_var[:,-generator.resp_dur:,:] - target_var[:,-generator.resp_dur:,:]), np.pi)) + l2_penalty

    # Create the update expressions
    params       = lasagne.layers.get_all_params(l_out, trainable=True)
    updates      = lasagne.updates.adam(loss, params, learning_rate=0.0005)

    # Compile the function for a training step, as well as the prediction function and a utility function to get the inner details of the RNN
    train_fn     = theano.function([input_var, target_var, mask_var], loss, updates=updates, allow_input_downcast=True)
    pred_fn      = theano.function([input_var, mask_var], pred_var, allow_input_downcast=True)
    rec_layer_fn = theano.function([input_var, mask_var], lasagne.layers.get_output(l_rec, get_details=True), allow_input_downcast=True)

    # TRAINING
    s_vec, opt_s_vec, ex_pred_vec, frac_rmse_vec = [], [], [], []
    for i, (_, example_input, example_output, example_mask, s, opt_s) in generator:
        score              = train_fn(example_input, example_output, example_mask)
        example_prediction = pred_fn(example_input, example_mask)
        s_vec.append(s)
        opt_s_vec.append(opt_s[:,-5,:])
        ex_pred_vec.append(example_prediction[:,-5,:])
        if i % 500 == 0:
            rmse_opt  = np.nanmean( np.mod(np.abs(np.asarray(s_vec) - np.asarray(opt_s_vec) ), np.pi) )
            rmse_net  = np.nanmean( np.mod(np.abs(np.squeeze(np.asarray(s_vec)) - np.squeeze(np.asarray(ex_pred_vec))), np.pi))
            frac_rmse = (rmse_net - rmse_opt) / rmse_opt
            frac_rmse_vec.append(frac_rmse)
            print 'Job_idx: %i; Batch: #%d; Frac. RMSE: %.6f; Opt. RMSE: %.6f; Net. RMSE: %.6f' % (job_idx, i, frac_rmse, rmse_opt, rmse_net)
            s_vec       = []
            opt_s_vec   = []
            ex_pred_vec = []

    # TESTING
    delay_vec, s_vec, opt_s_vec, ex_pred_vec, ex_hid_vec, ex_inp_vec = [], [], [], [], [], []
    for i, (delay_durs, example_input, example_output, example_mask, s, opt_s) in test_generator:
        example_prediction = pred_fn(example_input, example_mask)
        example_hidden     = rec_layer_fn(example_input, example_mask)
        s_vec.append(s)
        opt_s_vec.append(opt_s[:,-5,:])
        ex_pred_vec.append(example_prediction[:,-5,:])
        if i % 500 == 0:
            ex_hid_vec.append(example_hidden)
            ex_inp_vec.append(example_input)
            delay_vec.append(delay_durs)

    rmse_opt       = np.nanmean( np.mod(np.abs(np.asarray(s_vec) - np.asarray(opt_s_vec) ), np.pi) )
    rmse_net       = np.nanmean( np.mod(np.abs(np.squeeze(np.asarray(s_vec)) - np.squeeze(np.asarray(ex_pred_vec))), np.pi))
    frac_rmse_test = (rmse_net - rmse_opt) / rmse_opt
    print 'Test data; Frac. RMSE: %.6f; Opt. RMSE: %.6f; Net. RMSE: %.6f' % (frac_rmse_test, rmse_opt, rmse_net)

    ex_hid_vec = np.asarray(ex_hid_vec)
    ex_hid_vec = np.reshape(ex_hid_vec,(-1, test_generator.stim_dur + test_generator.max_delay + test_generator.resp_dur, n_hid))

    ex_inp_vec = np.asarray(ex_inp_vec)
    ex_inp_vec = np.reshape(ex_inp_vec,(-1, test_generator.stim_dur + test_generator.max_delay + test_generator.resp_dur, test_generator.n_loc * test_generator.n_in))

    # SAVE TRAINED MODEL
    sio.savemat('500_vardelay_DE2_sigma%f_lambda%f_everything.mat' % (offdiag_val, diag_val),
                {'all_params_list': lasagne.layers.get_all_param_values(l_out, trainable=True),
                 'inpResps': ex_inp_vec,
                 'hidResps': ex_hid_vec,
                 'frac_rmse_test': infloss_test,
                 'frac_rmse_vec': np.asarray(frac_rmse_vec),
                 'delay_vec': np.asarray(delay_vec)})