#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:19:34 2017 @author: emin
"""
import theano.tensor as T
import numpy as np
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init
import generators, models
from scipy.stats import entropy

def compute_SI(hidr, entrpy_bins, window_size, r_threshold):
    # Compute SI scores for a number of trials
    # hidr: activities of recurrent units
    # entrpy_bins: number of bins used for estimating peak time entropy
    # window_size: window size around peak time for calculating ridge-to-background ratio
    # r_threshold: only consider recurrent units with mean responses above this
    
    bs = hidr.shape[0]  # number of trials
    ts = hidr.shape[1]  # number of time points

    SI_trial_vec = np.zeros(bs)
    for b in range(bs):
        hidr_t = hidr[b, :, :]
        selected_indx = np.nonzero(np.mean(hidr_t, axis=0) > r_threshold)[0]
        hidr_t = hidr_t[:, selected_indx]

        peak_times = np.argmax(hidr_t, axis=0)
        end_times = np.clip(peak_times + window_size / 2, 0, ts - 1)
        start_times = np.clip(peak_times - window_size / 2, 0, ts - 1)
        entrpy = entropy(np.histogram(peak_times, entrpy_bins)[0] + 0.1 * np.ones(entrpy_bins))

        r2b_ratio = np.zeros(len(selected_indx))
        for nind in range(len(selected_indx)):
            mask = np.zeros(ts)
            mask[start_times[nind]:end_times[nind]] = 1
            this_hidr = hidr_t[:, nind]
            ridge = np.mean(this_hidr[start_times[nind]:end_times[nind]])
            backgr = np.mean(np.ma.MaskedArray(this_hidr, mask))
            r2b_ratio[nind] = np.log(ridge) - np.log(backgr)

        SI_trial_vec[b] = np.nanmean(r2b_ratio) + entrpy

    return np.nanmean(SI_trial_vec)

def build_generators(ExptDict):
    # Unpack common variables
    task          = ExptDict["task"]["task_id"]
    n_loc         = ExptDict["task"]["n_loc"]
    n_out         = ExptDict["task"]["n_out"]
    tr_cond       = ExptDict["tr_cond"]
    test_cond     = ExptDict["test_cond"]
    n_in          = ExptDict["n_in"]
    batch_size    = ExptDict["batch_size"]
    stim_dur      = ExptDict["stim_dur"]
    delay_dur     = ExptDict["delay_dur"]
    resp_dur      = ExptDict["resp_dur"]
    kappa         = ExptDict["kappa"]
    spon_rate     = ExptDict["spon_rate"]
    tr_max_iter   = ExptDict["tr_max_iter"]
    test_max_iter = ExptDict["test_max_iter"]   
    
    if task == 'DE1':
        generator = generators.DelayedEstimationTask(max_iter=tr_max_iter, 
                                                     batch_size=batch_size, 
                                                     n_loc=n_loc, n_in=n_in, 
                                                     n_out=n_out, stim_dur=stim_dur, 
                                                     delay_dur=delay_dur, 
                                                     resp_dur=resp_dur, kappa=kappa, 
                                                     spon_rate=spon_rate, 
                                                     tr_cond=tr_cond)   
                                                
        test_generator = generators.DelayedEstimationTask(max_iter=test_max_iter, 
                                                          batch_size=batch_size, 
                                                          n_loc=n_loc, n_in=n_in, 
                                                          n_out=n_out, 
                                                          stim_dur=stim_dur, 
                                                          delay_dur=delay_dur, 
                                                          resp_dur=resp_dur, 
                                                          kappa=kappa, 
                                                          spon_rate=spon_rate, 
                                                          tr_cond=test_cond)
    elif task == 'DE2':
        generator = generators.DelayedEstimationTask(max_iter=tr_max_iter, 
                                                     batch_size=batch_size, 
                                                     n_loc=n_loc, n_in=n_in, 
                                                     n_out=n_out, stim_dur=stim_dur, 
                                                     delay_dur=delay_dur, 
                                                     resp_dur=resp_dur, kappa=kappa, 
                                                     spon_rate=spon_rate, 
                                                     tr_cond=tr_cond)    
                                               
        test_generator = generators.DelayedEstimationTask(max_iter=test_max_iter, 
                                                          batch_size=batch_size, 
                                                          n_loc=n_loc, n_in=n_in, 
                                                          n_out=n_out, 
                                                          stim_dur=stim_dur, 
                                                          delay_dur=delay_dur, 
                                                          resp_dur=resp_dur, 
                                                          kappa=kappa, 
                                                          spon_rate=spon_rate, 
                                                          tr_cond=test_cond)
    elif task == 'CD1':
        generator = generators.ChangeDetectionTask(max_iter=tr_max_iter, 
                                                   batch_size=batch_size, 
                                                   n_loc=n_loc, n_in=n_in, 
                                                   n_out=n_out, stim_dur=stim_dur, 
                                                   delay_dur=delay_dur, 
                                                   resp_dur=resp_dur, kappa=kappa, 
                                                   spon_rate=spon_rate, 
                                                   tr_cond=tr_cond)
        
        test_generator = generators.ChangeDetectionTask(max_iter=test_max_iter, 
                                                        batch_size=batch_size, 
                                                        n_loc=n_loc, n_in=n_in, 
                                                        n_out=n_out, 
                                                        stim_dur=stim_dur, 
                                                        delay_dur=delay_dur, 
                                                        resp_dur=resp_dur, 
                                                        kappa=kappa, 
                                                        spon_rate=spon_rate, 
                                                        tr_cond=test_cond)
    elif task == 'COMP':
        generator = generators.ComparisonTask(max_iter=tr_max_iter,
                                                   batch_size=batch_size,
                                                   n_loc=n_loc, n_in=n_in,
                                                   n_out=n_out, stim_dur=stim_dur,
                                                   delay_dur=delay_dur,
                                                   resp_dur=resp_dur, sig_tc=10.0,
                                                   spon_rate=spon_rate,
                                                   tr_cond=tr_cond)

        test_generator = generators.ComparisonTask(max_iter=test_max_iter,
                                                        batch_size=batch_size,
                                                        n_loc=n_loc, n_in=n_in,
                                                        n_out=n_out,
                                                        stim_dur=stim_dur,
                                                        delay_dur=delay_dur,
                                                        resp_dur=resp_dur,
                                                        sig_tc=10.0,
                                                        spon_rate=spon_rate,
                                                        tr_cond=test_cond)
    elif task == 'CD2':
        generator = generators.ChangeDetectionTask(max_iter=tr_max_iter, 
                                                   batch_size=batch_size, 
                                                   n_loc=n_loc, n_in=n_in, 
                                                   n_out=n_out, stim_dur=stim_dur, 
                                                   delay_dur=delay_dur, 
                                                   resp_dur=resp_dur, kappa=kappa, 
                                                   spon_rate=spon_rate, 
                                                   tr_cond=tr_cond)
        
        test_generator = generators.ChangeDetectionTask(max_iter=test_max_iter, 
                                                        batch_size=batch_size, 
                                                        n_loc=n_loc, n_in=n_in, 
                                                        n_out=n_out, 
                                                        stim_dur=stim_dur, 
                                                        delay_dur=delay_dur, 
                                                        resp_dur=resp_dur, 
                                                        kappa=kappa, 
                                                        spon_rate=spon_rate, 
                                                        tr_cond=test_cond)
    elif task == 'GDE2':
        generator = generators.GatedDelayedEstimationTask(max_iter=tr_max_iter, 
                                                          batch_size=batch_size, 
                                                          n_loc=n_loc, n_in=n_in, 
                                                          n_out=n_out, 
                                                          stim_dur=stim_dur, 
                                                          delay_dur=delay_dur, 
                                                          resp_dur=resp_dur, 
                                                          kappa=kappa, 
                                                          spon_rate=spon_rate, 
                                                          tr_cond=tr_cond)
        
        test_generator = generators.GatedDelayedEstimationTask(max_iter=test_max_iter,
                                                               batch_size=batch_size, 
                                                               n_loc=n_loc, n_in=n_in,
                                                               n_out=n_out, 
                                                               stim_dur=stim_dur, 
                                                               delay_dur=delay_dur, 
                                                               resp_dur=resp_dur, 
                                                               kappa=kappa, 
                                                               spon_rate=spon_rate, 
                                                               tr_cond=test_cond)
    elif task == 'VDE1':
        max_delay = ExptDict["task"]["max_delay"]
        generator = generators.VarDelayedEstimationTask(max_iter=tr_max_iter, 
                                                        batch_size=batch_size, 
                                                        n_loc=n_loc, n_in=n_in, 
                                                        n_out=n_out, stim_dur=stim_dur, 
                                                        max_delay=max_delay, 
                                                        resp_dur=resp_dur, 
                                                        kappa=kappa, 
                                                        spon_rate=spon_rate, 
                                                        tr_cond=tr_cond)
        
        test_generator = generators.VarDelayedEstimationTask(max_iter=test_max_iter, 
                                                             batch_size=batch_size, 
                                                             n_loc=n_loc, n_in=n_in, 
                                                             n_out=n_out, 
                                                             stim_dur=stim_dur, 
                                                             max_delay=max_delay, 
                                                             resp_dur=resp_dur, 
                                                             kappa=kappa, 
                                                             spon_rate=spon_rate, 
                                                             tr_cond=test_cond)    
    elif task == 'Harvey2012':
        sigtc     = ExptDict["task"]["sigtc"]
        stim_rate = ExptDict["task"]["stim_rate"]
        generator = generators.Harvey2012(max_iter=tr_max_iter, batch_size=batch_size, 
                                          n_in=n_in, n_out=n_out, stim_dur=stim_dur, 
                                          delay_dur=delay_dur, resp_dur=resp_dur, 
                                          sigtc=sigtc, stim_rate=stim_rate, spon_rate=spon_rate)
        
        test_generator = generators.Harvey2012(max_iter=test_max_iter, batch_size=batch_size,
                                          n_in=n_in, n_out=n_out, stim_dur=stim_dur, 
                                          delay_dur=delay_dur, resp_dur=resp_dur, 
                                          sigtc=sigtc, stim_rate=stim_rate, spon_rate=spon_rate)

    elif task == 'Harvey2012Dynamic':
        sigtc     = ExptDict["task"]["sigtc"]
        stim_rate = ExptDict["task"]["stim_rate"]
        generator = generators.Harvey2012Dynamic(max_iter=tr_max_iter, batch_size=batch_size, 
                                          n_in=n_in, n_out=n_out, stim_dur=stim_dur, 
                                          delay_dur=delay_dur, resp_dur=resp_dur, 
                                          sigtc=sigtc, stim_rate=stim_rate, spon_rate=spon_rate)
        
        test_generator = generators.Harvey2012Dynamic(max_iter=test_max_iter, batch_size=batch_size,
                                          n_in=n_in, n_out=n_out, stim_dur=stim_dur, 
                                          delay_dur=delay_dur, resp_dur=resp_dur, 
                                          sigtc=sigtc, stim_rate=stim_rate, spon_rate=spon_rate)
        
    elif task == 'Harvey2016':
        sigtc     = ExptDict["task"]["sigtc"]
        stim_rate = ExptDict["task"]["stim_rate"]
        epoch_dur = ExptDict["task"]["epoch_dur"]
        n_epochs  = ExptDict["task"]["n_epochs"]
        generator = generators.Harvey2016(max_iter=tr_max_iter, batch_size=batch_size, 
                                          n_in=n_in, n_out=n_out, n_epochs=n_epochs, 
                                          epoch_dur=epoch_dur, sigtc=sigtc, 
                                          stim_rate=stim_rate, spon_rate=spon_rate)
        
        test_generator = generators.Harvey2016(max_iter=test_max_iter, batch_size=batch_size,
                                          n_in=n_in, n_out=n_out, n_epochs=n_epochs, 
                                          epoch_dur=epoch_dur, sigtc=sigtc, 
                                          stim_rate=stim_rate, spon_rate=spon_rate)

    elif task == 'SINE':
        alpha     = ExptDict["task"]["alpha"]
        generator = generators.SineTask(max_iter=tr_max_iter, batch_size=batch_size, 
                                          n_in=n_in, n_out=n_out, stim_dur=stim_dur,
                                          delay_dur=delay_dur, resp_dur=resp_dur, alpha=alpha)
        
        test_generator = generators.SineTask(max_iter=test_max_iter, batch_size=batch_size, 
                                          n_in=n_in, n_out=n_out, stim_dur=stim_dur,
                                          delay_dur=delay_dur, resp_dur=resp_dur, alpha=alpha)
        
    return generator, test_generator 

def build_model(input_var,ExptDict):
    # Unpack necessary variables
    model      = ExptDict["model"]["model_id"]
    n_loc      = ExptDict["task"]["n_loc"]
    n_out      = ExptDict["task"]["n_out"]
    batch_size = ExptDict["batch_size"]
    n_in       = ExptDict["n_in"]
    n_hid      = ExptDict["n_hid"]
    out_nonlin = ExptDict["task"]["out_nonlin"]
    
    if model == 'LeInitRecurrent':  
        diag_val     = ExptDict["model"]["diag_val"]
        offdiag_val  = ExptDict["model"]["offdiag_val"]
        l_out, l_rec = models.LeInitRecurrent(input_var, batch_size=batch_size, 
                                              n_in=n_loc*n_in, n_out=n_out, 
                                              n_hid=n_hid, diag_val=diag_val,
                                              offdiag_val=offdiag_val,
                                              out_nlin=out_nonlin)
    elif model == 'LeInitRecurrentWithFastWeights':  
        diag_val     = ExptDict["model"]["diag_val"]
        offdiag_val  = ExptDict["model"]["offdiag_val"]
        gamma        = ExptDict["model"]["gamma"]
        l_out, l_rec = models.LeInitRecurrentWithFastWeights(input_var, batch_size=batch_size, 
                                              n_in=n_loc*n_in, n_out=n_out, 
                                              n_hid=n_hid, diag_val=diag_val,
                                              offdiag_val=offdiag_val,
                                              out_nlin=out_nonlin, gamma=gamma)
    elif model == 'OrthoInitRecurrent':  
        init_val = ExptDict["model"]["init_val"]
        l_out, l_rec = models.OrthoInitRecurrent(input_var, batch_size=batch_size, 
                                              n_in=n_loc*n_in, n_out=n_out, 
                                              n_hid=n_hid, init_val=init_val, 
                                              out_nlin=out_nonlin)
    elif model == 'GRURecurrent':
        diag_val     = ExptDict["model"]["diag_val"]
        offdiag_val  = ExptDict["model"]["offdiag_val"]
        l_out, l_rec = models.GRURecurrent(input_var, batch_size=batch_size, 
                                           n_in=n_loc*n_in, n_out=n_out, n_hid=n_hid, 
                                           diag_val=diag_val, offdiag_val=offdiag_val,
                                           out_nlin=out_nonlin)
        
    return l_out, l_rec

def build_loss(pred_var,target_var,ExptDict):
    # Unpack necessary variables
    task     = ExptDict["task"]["task_id"]
    resp_dur = ExptDict["resp_dur"]
    
    if task in ['DE1','DE2','GDE2','VDE1']:
        loss = T.mean(T.mod(T.abs_(pred_var[:,-resp_dur:,:] - target_var[:,-resp_dur:,:]), np.pi))
    elif task in ['CD1','CD2','Harvey2012','Harvey2012Dynamic','Harvey2016','COMP']:
        loss = T.mean(lasagne.objectives.binary_crossentropy(pred_var[:,-resp_dur:,-1], target_var[:,-resp_dur:,-1])) 
    elif task in ['SINE']:
        loss = T.mean(T.abs_(pred_var[:,-resp_dur:,:] - target_var[:,-resp_dur:,:]))

    return loss

def build_performance(s_vec,opt_vec,net_vec,ExptDict):    
    # Unpack necessary variables
    task     = ExptDict["task"]["task_id"]

    if task in ['DE1','DE2','GDE2','VDE1']:
        rmse_opt = np.nanmean(np.mod(np.abs(np.squeeze(s_vec) - np.squeeze(opt_vec)), np.pi)) 
        rmse_net = np.nanmean(np.mod(np.abs(np.squeeze(s_vec) - np.squeeze(net_vec)), np.pi))
        infloss  = (rmse_net - rmse_opt) / rmse_opt
    elif task in ['CD1','CD2','Harvey2012','Harvey2012Dynamic','Harvey2016','COMP']:
        infloss = np.nanmean( opt_vec * np.log(opt_vec/net_vec) + (1.0 - opt_vec) * np.log((1.0 - opt_vec)/(1.0 - net_vec)) ) / np.nanmean( opt_vec * np.log(2.0*opt_vec) + (1.0-opt_vec) * np.log(2.0*(1.0-opt_vec)) ) 
    elif task in ['SINE']:
        infloss = np.mean(np.abs(np.squeeze(opt_vec) - np.squeeze(net_vec)))
    return infloss
