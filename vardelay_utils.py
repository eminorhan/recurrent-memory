#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:19:34 2017 @author: emin
"""
import theano.tensor as T
import numpy as np
import generators, models
from lasagne.objectives import binary_crossentropy

def build_generators(t_ind):
    if t_ind==0:
        generator = generators.VarDelayedEstimationTask(max_iter=25001, batch_size=50, n_loc=1, n_in=50, n_out=1,
                                                        stim_dur=25, max_delay=100, resp_dur=25, kappa=2.0,
                                                        spon_rate=0.1, tr_cond='all_gains')
        test_generator = generators.VarDelayedEstimationTask(max_iter=2501, batch_size=50, n_loc=1, n_in=50, n_out=1,
                                                             stim_dur=25, max_delay=100, resp_dur=25, kappa=2.0,
                                                             spon_rate=0.1, tr_cond='all_gains')
    elif t_ind==1:
        generator = generators.VarDelayedEstimationTask(max_iter=25001, batch_size=50,  n_loc=2, n_in=50, n_out=2,
                                                        stim_dur=25, max_delay=100, resp_dur=25, kappa=2.0,
                                                        spon_rate=0.1, tr_cond='all_gains')
        test_generator = generators.VarDelayedEstimationTask(max_iter=2501, batch_size=50, n_loc=2, n_in=50, n_out=2,
                                                             stim_dur=25, max_delay=100, resp_dur=25, kappa=2.0,
                                                             spon_rate=0.1, tr_cond='all_gains')
    elif t_ind==2:
        generator = generators.VarChangeDetectionTask(max_iter=25001, batch_size=50, n_loc=1, n_in=50, n_out=1,
                                                      stim_dur=25, max_delay=100, resp_dur=25, kappa=2.0,
                                                      spon_rate=0.1, tr_cond='all_gains')
        test_generator = generators.VarChangeDetectionTask(max_iter=2501, batch_size=50, n_loc=1, n_in=50, n_out=1,
                                                           stim_dur=25, max_delay=100, resp_dur=25, kappa=2.0,
                                                           spon_rate=0.1, tr_cond='all_gains')
    elif t_ind==4:
        generator = generators.VarGatedDelayedEstimationTask(max_iter=25001, batch_size=50, n_loc=2, n_in=50, n_out=1,
                                                             stim_dur=25, max_delay=100, resp_dur=25, kappa=2.0,
                                                             spon_rate=0.1, tr_cond='all_gains')
        test_generator = generators.VarGatedDelayedEstimationTask(max_iter=2501, batch_size=50, n_loc=2, n_in=50, n_out=1,
                                                                  stim_dur=25, max_delay=100, resp_dur=25, kappa=2.0,
                                                                  spon_rate=0.1, tr_cond='all_gains')
    elif t_ind==6:
        generator  = generators.VarHarvey2012(max_iter=25001, batch_size=50, n_in=50, n_out=1, stim_dur=25, max_delay=100,
                                              resp_dur=25, sigtc=15.0, stim_rate=1.0, spon_rate=0.1)
        test_generator = generators.VarHarvey2012(max_iter=2501, batch_size=50, n_in=50, n_out=1, stim_dur=25, max_delay=100,
                                                  resp_dur=25, sigtc=15.0, stim_rate=1.0, spon_rate=0.1)
    elif t_ind==8:
        generator = generators.VarComparisonTask(max_iter=25001, batch_size=50, n_loc=1, n_in=50, n_out=1, stim_dur=25,
                                                 max_delay=100, resp_dur=25, sig_tc=10.0, spon_rate=0.1, tr_cond='all_gains')
        test_generator = generators.VarComparisonTask(max_iter=2501, batch_size=50, n_loc=1, n_in=50, n_out=1, stim_dur=25,
                                                      max_delay=100, resp_dur=25, sig_tc=10.0, spon_rate=0.1, tr_cond='all_gains')
    return generator, test_generator

def build_loss(pred_var, target_var, resp_dur, t_ind):
    if t_ind==0 or t_ind==1 or t_ind==4:
        loss = T.mean(T.mod(T.abs_(pred_var[:, -resp_dur:, :] - target_var[:, -resp_dur:, :]), np.pi))
    elif t_ind==2 or t_ind==6 or t_ind==8:
        loss = T.mean(binary_crossentropy(pred_var[:,-resp_dur:,-1], target_var[:,-resp_dur:,-1]))
    return loss

def build_performance(s_vec, opt_vec, net_vec, t_ind):
    if t_ind==0 or t_ind==1 or t_ind==4:
        rmse_opt = np.nanmean(np.mod(np.abs(s_vec - opt_s_vec), np.pi))
        rmse_net = np.nanmean( np.mod(np.abs(np.squeeze(s_vec) - np.squeeze(ex_pred_vec)), np.pi))

        performance = (rmse_net - rmse_opt) / rmse_opt
    elif t_ind==2 or t_ind==6 or t_ind==8:
        performance = np.nanmean(opt_vec * np.log(opt_vec/net_vec) + (1.0 - opt_vec) * np.log((1.0 - opt_vec)/(1.0 - net_vec)) )\
                      / np.nanmean( opt_vec * np.log(2.0*opt_vec) + (1.0-opt_vec) * np.log(2.0*(1.0-opt_vec)) )
    return performance