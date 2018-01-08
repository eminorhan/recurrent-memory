#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:20:09 2017 @author: emin
"""
import os
import numpy as np 
from scipy.io import loadmat,savemat
from scipy.stats import entropy

# --- Calculates sequentiality index defined as Entropy[peak resp time dist] + Mean log(R2B ratio)
n_lambda          = 10
n_offdiag         = 10
n_jobs            = n_lambda * n_offdiag
r2b_means         = np.zeros(n_jobs)
inflosses         = np.zeros(n_jobs)
entrpy_means      = np.zeros(n_jobs)
infloss_threshold = 0.5       # only consider jobs with inf. loss below this
entrpy_bins       = 20        # number of bins for computing entropy of peak resp. time distribution
window_size       = 5         # window around peak time to compute mean ridge activity
r_threshold       = 1e-1      # only consider neurons with mean activity above this

# CD1: task 2
# DE1: task 0
# DE2: task 1
# GDE2: task 4
# HAR12: task 6
# COMP: task 8

SI_job_vec = np.zeros(n_jobs)
for j in range(n_jobs):

    filename = 'DATA/BASIC_COMP_data/500jobidx%i_model0_task8_everything.mat'%(j+1)
    if os.path.isfile(filename):
        data = loadmat(filename)
    else:
        continue

    hidr = data['hidResps'] # batch_size x time x nneuron
    infloss_test = data['infloss_test']
#    infloss_test = data['frac_rmse_test']    
    
    print 'Starting job %i'%(j+1)
    if (infloss_test<infloss_threshold):
        bs = hidr.shape[0] # number of trials
        ts = hidr.shape[1] # number of time points
        ns = hidr.shape[2] # number of neurons
        
        SI_trial_vec = np.zeros(bs)
        for b in range(bs):
            hidr_t = hidr[b,:,:]
            selected_indx = np.nonzero(np.mean(hidr_t,axis=0)>r_threshold)[0]
            hidr_t = hidr_t[:,selected_indx]
            
            peak_times = np.argmax(hidr_t,axis=0)
            end_times = np.clip(peak_times + window_size/2, 0, ts-1)
            start_times = np.clip(peak_times - window_size/2, 0, ts-1)
            entrpy = entropy(np.histogram(peak_times, entrpy_bins)[0]+0.1*np.ones(entrpy_bins))
            
            r2b_ratio = np.zeros(len(selected_indx))
            for nind in range(len(selected_indx)):
                mask = np.zeros(ts)
                mask[start_times[nind]:end_times[nind]] = 1
                this_hidr = hidr_t[:,nind]
                ridge = np.mean(this_hidr[start_times[nind]:end_times[nind]])
                backgr = np.mean(np.ma.MaskedArray(this_hidr,mask))
                r2b_ratio[nind] = np.log(ridge) - np.log(backgr)
            
            SI_trial_vec[b] = np.nanmean(r2b_ratio) + entrpy # compute SI for this trial
                        
        SI_job_vec[j] = np.nanmean(SI_trial_vec)
        print 'Mean SI %f'%SI_job_vec[j]

savemat('BASIC_COMP_SI.mat',{'SI':SI_job_vec})
