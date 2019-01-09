# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:53:50 2016 by @author: emin
"""
import numpy as np
from scipy.misc import comb
import scipy.stats as scistat

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the given axis
    """
    b        = np.random.random(a.shape)
    idx      = np.argsort(b, axis=axis)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled
    
class Task(object):

    def __init__(self, max_iter=None, batch_size=1):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1) , self.sample()
        else:
            raise StopIteration()

    def sample(self):
        raise NotImplementedError()


class Harvey2012(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_in=25, n_out=1, stim_dur=50, delay_dur=50, resp_dur=10, sigtc=10.0, stim_rate=1.0, spon_rate=0.1):
        super(Harvey2012, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                             
        self.n_out     = n_out
        self.tau       = 1.0 / sigtc**2
        self.spon_rate = spon_rate
        self.phi       = np.linspace(-40.0, 40.0, self.n_in) 
        self.stim_dur  = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.stim_rate = stim_rate
        
    def sample(self):                        
        # Left-right choice         
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size,))
        S              = -15.0 * (C==0.0) + 15.0 * (C==1.0)
                
        # Noisy responses
        Ls             = (self.stim_rate / self.stim_dur) * np.exp(-0.5 * self.tau * ( np.tile(np.swapaxes(np.tile(S, (1,1,1)), 0,2),(1, self.stim_dur, self.n_in)) - np.tile(self.phi,(self.batch_size,self.stim_dur,1) ) )**2 )
        Ld             = (self.spon_rate / self.delay_dur) * np.ones((self.batch_size, self.delay_dur, self.n_in)) 
        Lr             = (self.spon_rate / self.resp_dur) * np.ones((self.batch_size, self.resp_dur, self.n_in))

        Rs             = np.random.poisson(Ls)
        Rd             = np.random.poisson(Ld)
        Rr             = np.random.poisson(Lr)

        example_input  = np.concatenate((Rs,Rd,Rr), axis=1)
        example_output = np.repeat(C[:,np.newaxis],self.total_dur,axis=1)
        example_output = np.repeat(example_output[:,:,np.newaxis],1,axis=2)

        cum_Rs         = np.sum(Rs,axis=1)
        prec           = np.sum(cum_Rs,axis=1) * self.tau
        mu             = self.tau * np.dot(cum_Rs,self.phi) / prec
        d              = 0.5 * prec * ( (-15.0 - mu)**2 - (15.0 - mu)**2 ) 
        P1             = 1.0 / (1.0 + np.exp(-d))

        return example_input, example_output, C, P1


class VarHarvey2012(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_in=25, n_out=1, stim_dur=50, max_delay=50, resp_dur=10, sigtc=10.0, stim_rate=1.0, spon_rate=0.1):
        super(VarHarvey2012, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                             
        self.n_out     = n_out
        self.tau       = 1.0 / sigtc**2
        self.spon_rate = spon_rate
        self.phi       = np.linspace(-40.0, 40.0, self.n_in) 
        self.stim_dur  = stim_dur
        self.max_delay = max_delay
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + max_delay + resp_dur
        self.stim_rate = stim_rate
        
    def sample(self):                        
        # Left-right choice         
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size,))
        S              = -15.0 * (C==0.0) + 15.0 * (C==1.0)
        delay_durs     = np.random.choice([10,40,70,100], size=(self.batch_size,))
                
        # Noisy responses
        Ls             = (self.stim_rate / self.stim_dur) * np.exp(-0.5 * self.tau * ( np.tile(np.swapaxes(np.tile(S, (1,1,1)), 0,2),(1, self.stim_dur, self.n_in)) - np.tile(self.phi,(self.batch_size,self.stim_dur,1) ) )**2 )
        Ld             = (self.spon_rate / self.max_delay) * np.ones((self.batch_size, self.max_delay, self.n_in)) 
        Lr             = (self.spon_rate / self.resp_dur) * np.ones((self.batch_size, self.resp_dur, self.n_in))

        Rs             = np.random.poisson(Ls)
        Rd             = np.random.poisson(Ld)
        Rr             = np.random.poisson(Lr)

        example_input  = np.concatenate((Rs,Rd,Rr), axis=1)
        example_output = np.repeat(C[:,np.newaxis],self.total_dur,axis=1)
        example_output = np.repeat(example_output[:,:,np.newaxis],1,axis=2)
        # example_output[:,-5:,:] = 0.5
        example_mask   = np.ones((self.batch_size,self.total_dur), dtype=np.bool)
        for i in range(self.batch_size):
            example_mask[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay)] = False
            example_input[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0
            example_output[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0        
        
        cum_Rs         = np.sum(Rs,axis=1)
        prec           = np.sum(cum_Rs,axis=1) * self.tau
        mu             = self.tau * np.dot(cum_Rs,self.phi) / prec
        d              = 0.5 * prec * ( (-15.0 - mu)**2 - (15.0 - mu)**2 ) 
        P1             = 1.0 / (1.0 + np.exp(-d))

        return delay_durs, example_input, example_output, example_mask, C, P1
 

class Harvey2012Dynamic(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_in=25, n_out=1, stim_dur=50, delay_dur=50, resp_dur=10, sigtc=10.0, stim_rate=1.0, spon_rate=0.1):
        super(Harvey2012Dynamic, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                             
        self.n_out     = n_out
        self.tau       = 1.0 / sigtc**2
        self.spon_rate = spon_rate
        self.stim_dur  = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.stim_rate = stim_rate
        self.phi       = np.linspace(0.0, self.total_dur - 1.0, self.n_in/2) 
        self.S         = np.linspace(0.0, self.total_dur - 1.0, self.total_dur)
        self.SS,self.PP= np.meshgrid(self.S,self.phi)
        self.E         = np.exp(-0.5 * self.tau * (self.SS-self.PP)**2).T
        self.E         = np.tile(self.E,(self.batch_size,1,1)) 
        
    def sample(self):                        
        # Left-right choice 
        C      = np.random.choice([0.0, 1.0], size=(self.batch_size,))
        gs1    = (C==0.00) * self.spon_rate + (C==1.0) * self.stim_rate
        gs2    = (C==1.00) * self.spon_rate + (C==0.0) * self.stim_rate
        gs1    = np.repeat(gs1[:,np.newaxis],self.stim_dur,axis=1)  
        gs1    = np.repeat(gs1[:,:,np.newaxis],self.n_in/2,axis=2)
        gs2    = np.repeat(gs2[:,np.newaxis],self.stim_dur,axis=1)        
        gs2    = np.repeat(gs2[:,:,np.newaxis],self.n_in/2,axis=2)       

        # Noisy responses
        Ls     = (1.0 / self.stim_dur) * np.concatenate((gs1,gs2),axis=2)
        Ld     = (self.spon_rate / self.delay_dur) * np.ones((self.batch_size, self.delay_dur, self.n_in)) 
        Lr     = (1.0 * self.spon_rate / self.resp_dur) * np.ones((self.batch_size, self.resp_dur, self.n_in))
        L      = np.concatenate((Ls,Ld,Lr),axis=1)

        example_input  = np.random.poisson(L*np.concatenate((self.E,self.E),axis=2))
        example_output = np.repeat(C[:,np.newaxis],self.total_dur,axis=1)
        example_output = np.repeat(example_output[:,:,np.newaxis],1,axis=2)
        example_output[:,-10:,:] = 0.5
                      
        n1 = np.sum(example_input[:,:self.stim_dur,:self.n_in/2],axis=1)
        n1 = np.sum(n1,axis=1)
        n2 = np.sum(example_input[:,:self.stim_dur,self.n_in/2:],axis=1)
        n2 = np.sum(n2,axis=1)
        m  = n2 - n1
        s  = 1./np.sqrt(n1 + n2)
        P1 = scistat.norm.cdf(0.,m,s)

        return example_input, example_output, C, P1
       

class ComparisonTask(Task):
    '''Parameters'''

    def __init__(self, max_iter=None, batch_size=50, n_loc=1, n_in=25, n_out=1, stim_dur=10, delay_dur=100, resp_dur=10,
                 sig_tc=10.0, spon_rate=0.001, tr_cond='all_gains'):
        super(ComparisonTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in = n_in  # number of neurons per location
        self.n_out = n_out
        self.n_loc = n_loc
        self.sig_tc = sig_tc
        self.spon_rate = spon_rate
        self.nneuron = self.n_in * self.n_loc  # total number of input neurons
        self.phi = np.linspace(-50.0, 50.0, self.n_in)  # Tuning curve centers
        self.stim_dur = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.tr_cond = tr_cond

    def sample(self):
        if self.tr_cond == 'all_gains':
            G = (1.0 / self.stim_dur) * np.random.choice([1.0], size=(self.n_loc, self.batch_size))
            G = np.repeat(G, self.n_in, axis=0).T
            G = np.tile(G, (self.stim_dur, 1, 1))
            G = np.swapaxes(G, 0, 1)
        else:
            G = (1.0 / self.stim_dur) * np.random.choice([1.0], size=(1, self.batch_size))
            G = np.repeat(G, self.n_in * self.n_loc, axis=0).T
            G = np.tile(G, (self.stim_dur, 1, 1))
            G = np.swapaxes(G, 0, 1)

        H = (1.0 / self.resp_dur) * np.ones((self.batch_size, self.resp_dur, self.nneuron))

        # Stimuli
        S1 = 80.0 * np.random.rand(self.n_loc, self.batch_size) - 40.0   # first stimulus
        S2 = 80.0 * np.random.rand(self.n_loc, self.batch_size) - 40.0   # second stimulus

        # Larger/smaller indicator
        C = (S1>S2).flatten() + 0.0

        S1 = np.repeat(S1, self.n_in, axis=0).T
        S1 = np.tile(S1, (self.stim_dur, 1, 1))
        S1 = np.swapaxes(S1, 0, 1)

        S2 = np.repeat(S2, self.n_in, axis=0).T
        S2 = np.tile(S2, (self.resp_dur, 1, 1))
        S2 = np.swapaxes(S2, 0, 1)

        # Noisy responses
        L1 = G * np.exp(-(0.5/self.sig_tc**2) * (S1 - np.tile(self.phi, (self.batch_size, self.stim_dur, self.n_loc)) )**2 )
        L2 = H * np.exp(-(0.5/self.sig_tc**2) * (S2 - np.tile(self.phi, (self.batch_size, self.resp_dur, self.n_loc)) )**2 )
        Ld = (self.spon_rate / self.delay_dur) * np.ones((self.batch_size, self.delay_dur, self.nneuron))  # delay

        R1 = np.random.poisson(L1)
        R2 = np.random.poisson(L2)
        Rd = np.random.poisson(Ld)

        example_input = np.concatenate((R1, Rd, R2), axis=1)
        example_output = np.repeat(C[:, np.newaxis], self.total_dur, axis=1)
        example_output = np.repeat(example_output[:, :, np.newaxis], 1, axis=2)

        cum_R1 = np.sum(R1, axis=1)
        cum_R2 = np.sum(R2, axis=1)

        mu_x = np.dot(cum_R1, self.phi) / np.sum(cum_R1, axis=1)
        mu_y = np.dot(cum_R2, self.phi) / np.sum(cum_R2, axis=1)

        v_x = self.sig_tc**2 / np.sum(cum_R1, axis=1)
        v_y = self.sig_tc**2 / np.sum(cum_R2, axis=1)

        if self.n_loc == 1:
            d = scistat.norm.cdf(0.0, mu_y-mu_x, np.sqrt(v_x+v_y))
        else:
            d = scistat.norm.cdf(0.0, mu_y-mu_x, np.sqrt(v_x+v_y))
        P = d
        return example_input, example_output, C, P


class VarComparisonTask(Task):
    '''Parameters'''

    def __init__(self, max_iter=None, batch_size=50, n_loc=1, n_in=25, n_out=1, stim_dur=10, max_delay=100, resp_dur=10,
                 sig_tc=10.0, spon_rate=0.001, tr_cond='all_gains'):
        super(VarComparisonTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in = n_in  # number of neurons per location
        self.n_out = n_out
        self.n_loc = n_loc
        self.sig_tc = sig_tc
        self.spon_rate = spon_rate
        self.nneuron = self.n_in * self.n_loc  # total number of input neurons
        self.phi = np.linspace(-50.0, 50.0, self.n_in)  # Tuning curve centers
        self.stim_dur = stim_dur
        self.max_delay = max_delay
        self.resp_dur = resp_dur
        self.total_dur = stim_dur + max_delay + resp_dur
        self.tr_cond = tr_cond

    def sample(self):
        if self.tr_cond == 'all_gains':
            G = (1.0 / self.stim_dur) * np.random.choice([1.0], size=(self.n_loc, self.batch_size))
            G = np.repeat(G, self.n_in, axis=0).T
            G = np.tile(G, (self.stim_dur, 1, 1))
            G = np.swapaxes(G, 0, 1)
        else:
            G = (1.0 / self.stim_dur) * np.random.choice([1.0], size=(1, self.batch_size))
            G = np.repeat(G, self.n_in * self.n_loc, axis=0).T
            G = np.tile(G, (self.stim_dur, 1, 1))
            G = np.swapaxes(G, 0, 1)

        H = (1.0 / self.resp_dur) * np.ones((self.batch_size, self.resp_dur, self.nneuron))

        delay_durs = np.random.choice([10,40,70,100], size=(self.batch_size,))

        # Stimuli
        S1 = 80.0 * np.random.rand(self.n_loc, self.batch_size) - 40.0   # first stimulus
        S2 = 80.0 * np.random.rand(self.n_loc, self.batch_size) - 40.0   # second stimulus

        # Larger/smaller indicator
        C = (S1>S2).flatten() + 0.0

        S1 = np.repeat(S1, self.n_in, axis=0).T
        S1 = np.tile(S1, (self.stim_dur, 1, 1))
        S1 = np.swapaxes(S1, 0, 1)

        S2 = np.repeat(S2, self.n_in, axis=0).T
        S2 = np.tile(S2, (self.resp_dur, 1, 1))
        S2 = np.swapaxes(S2, 0, 1)

        # Noisy responses
        L1 = G * np.exp(-(0.5/self.sig_tc**2) * (S1 - np.tile(self.phi, (self.batch_size, self.stim_dur, self.n_loc)) )**2 )
        L2 = H * np.exp(-(0.5/self.sig_tc**2) * (S2 - np.tile(self.phi, (self.batch_size, self.resp_dur, self.n_loc)) )**2 )
        Ld = (self.spon_rate / self.max_delay) * np.ones((self.batch_size, self.max_delay, self.nneuron))  # delay

        R1 = np.random.poisson(L1)
        R2 = np.random.poisson(L2)
        Rd = np.random.poisson(Ld)

        example_input = np.concatenate((R1, Rd, R2), axis=1)
        example_output = np.repeat(C[:, np.newaxis], self.total_dur, axis=1)
        example_output = np.repeat(example_output[:, :, np.newaxis], 1, axis=2)
        example_mask   = np.ones((self.batch_size,self.total_dur), dtype=np.bool)
        for i in range(self.batch_size):
            example_mask[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay)] = False
            example_input[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0
            example_output[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0

        cum_R1 = np.sum(R1, axis=1)
        cum_R2 = np.sum(R2, axis=1)

        mu_x = np.dot(cum_R1, self.phi) / np.sum(cum_R1, axis=1)
        mu_y = np.dot(cum_R2, self.phi) / np.sum(cum_R2, axis=1)

        v_x = self.sig_tc**2 / np.sum(cum_R1, axis=1)
        v_y = self.sig_tc**2 / np.sum(cum_R2, axis=1)

        if self.n_loc == 1:
            d = scistat.norm.cdf(0.0, mu_y-mu_x, np.sqrt(v_x+v_y))
        else:
            d = scistat.norm.cdf(0.0, mu_y-mu_x, np.sqrt(v_x+v_y))
        P = d
        return delay_durs, example_input, example_output, example_mask, C, P


class ChangeDetectionTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_loc=2, n_in=25, n_out=1, stim_dur=10, delay_dur=100, resp_dur=10, kappa=2.0, spon_rate=0.001, tr_cond='all_gains'):
        super(ChangeDetectionTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                             # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.spon_rate = spon_rate
        self.nneuron   = self.n_in * self.n_loc           # total number of input neurons
        self.phi       = np.linspace(0, np.pi, self.n_in) # np.linspace(0, np.pi, self.n_in)
        self.stim_dur  = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.tr_cond   = tr_cond
        
    def sample(self):            
        if self.tr_cond == 'all_gains':
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        
        H = (1.0/self.resp_dur) * np.ones((self.batch_size,self.resp_dur,self.nneuron))
        
        # Target presence/absence and stimuli 
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size,))
        C1ind          = np.where(C==1.0)[0]        # change

        S1             = np.pi * np.random.rand(self.n_loc, self.batch_size)
        S2             = S1
        S1             = np.repeat(S1,self.n_in,axis=0).T
        S1             = np.tile(S1,(self.stim_dur,1,1))
        S1             = np.swapaxes(S1,0,1)

        S2[np.random.randint(0,self.n_loc,size=(len(C1ind),)), C1ind] = np.pi * np.random.rand(len(C1ind))
        S2             = np.repeat(S2,self.n_in,axis=0).T
        S2             = np.tile(S2,(self.resp_dur,1,1))
        S2             = np.swapaxes(S2,0,1)
                
        # Noisy responses
        L1             = G * np.exp( self.kappa * (np.cos( 2.0 * (S1 - np.tile(self.phi, (self.batch_size,self.stim_dur,self.n_loc) ) ) ) - 1.0) ) # stim 1
        L2             = H * np.exp( self.kappa * (np.cos( 2.0 * (S2 - np.tile(self.phi, (self.batch_size,self.resp_dur,self.n_loc) ) ) ) - 1.0) ) # stim 2
        Ld             = (self.spon_rate / self.delay_dur) * np.ones((self.batch_size,self.delay_dur,self.nneuron))                                                   # delay

        R1             = np.random.poisson(L1)
        R2             = np.random.poisson(L2)
        Rd             = np.random.poisson(Ld)

        example_input  = np.concatenate((R1,Rd,R2), axis=1)
        example_output = np.repeat(C[:,np.newaxis],self.total_dur,axis=1)
        example_output = np.repeat(example_output[:,:,np.newaxis],1,axis=2)
        
        cum_R1         = np.sum(R1,axis=1) 
        cum_R2         = np.sum(R2,axis=1) 
        
        mu_x           = np.asarray([ np.arctan2( np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ])
        mu_y           = np.asarray([ np.arctan2( np.dot(cum_R2[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R2[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ])
        
        temp_x         = np.asarray([np.swapaxes(np.multiply.outer(cum_R1,cum_R1),1,2)[i,i,:,:] for i in range(self.batch_size)])
        temp_y         = np.asarray([np.swapaxes(np.multiply.outer(cum_R2,cum_R2),1,2)[i,i,:,:] for i in range(self.batch_size)])
        
        kappa_x        = np.asarray( [np.sqrt(np.sum(temp_x[:,i*self.n_in:(i+1)*self.n_in,i*self.n_in:(i+1)*self.n_in] * np.repeat(np.cos( np.subtract(np.expand_dims(self.phi,axis=1), np.expand_dims(self.phi,axis=1).T) )[np.newaxis,:,:],self.batch_size,axis=0),axis=(1,2))) for i in range(self.n_loc) ] )
        kappa_y        = np.asarray( [np.sqrt(np.sum(temp_y[:,i*self.n_in:(i+1)*self.n_in,i*self.n_in:(i+1)*self.n_in] * np.repeat(np.cos( np.subtract(np.expand_dims(self.phi,axis=1), np.expand_dims(self.phi,axis=1).T) )[np.newaxis,:,:],self.batch_size,axis=0),axis=(1,2))) for i in range(self.n_loc) ] )
        
        if self.n_loc==1:
            d          = np.i0(kappa_x) * np.i0(kappa_y) / np.i0( np.sqrt(kappa_x ** 2 + kappa_y ** 2 + 2.0 * kappa_x * kappa_y * np.cos(mu_y-mu_x)) )
        else:
            d          = np.nanmean(np.i0(kappa_x) * np.i0(kappa_y) / np.i0( np.sqrt(kappa_x ** 2 + kappa_y ** 2 + 2.0 * kappa_x * kappa_y * np.cos(mu_y-mu_x)) ), axis=0)
        
        P              = d / (d + 1.0)
        return example_input, example_output, C, P
    

class VarChangeDetectionTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_loc=2, n_in=25, n_out=1, stim_dur=10, max_delay=100, resp_dur=10, kappa=2.0, spon_rate=0.001, tr_cond='all_gains'):
        super(VarChangeDetectionTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                             # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.spon_rate = spon_rate
        self.nneuron   = self.n_in * self.n_loc           # total number of input neurons
        self.phi       = np.linspace(0, np.pi, self.n_in) # np.linspace(0, np.pi, self.n_in)
        self.stim_dur  = stim_dur
        self.max_delay = max_delay
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + max_delay + resp_dur
        self.tr_cond   = tr_cond
        
    def sample(self):            
        if self.tr_cond == 'all_gains':
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        
        H = (1.0/self.resp_dur) * np.ones((self.batch_size,self.resp_dur,self.nneuron)) 
        
        # Target presence/absence and stimuli 
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size,))
        C1ind          = np.where(C==1.0)[0]        # change

        delay_durs     = np.random.choice([10,40,70,100], size=(self.batch_size,))
        S1             = np.pi * np.random.rand(self.n_loc, self.batch_size)
        S2             = S1
        S1             = np.repeat(S1,self.n_in,axis=0).T
        S1             = np.tile(S1,(self.stim_dur,1,1))
        S1             = np.swapaxes(S1,0,1)

        S2[np.random.randint(0,self.n_loc,size=(len(C1ind),)), C1ind] = np.pi * np.random.rand(len(C1ind))
        S2             = np.repeat(S2,self.n_in,axis=0).T
        S2             = np.tile(S2,(self.resp_dur,1,1))
        S2             = np.swapaxes(S2,0,1)
                
        # Noisy responses
        L1             = G * np.exp( self.kappa * (np.cos( 2.0 * (S1 - np.tile(self.phi, (self.batch_size,self.stim_dur,self.n_loc) ) ) ) - 1.0) ) # stim 1
        L2             = H * np.exp( self.kappa * (np.cos( 2.0 * (S2 - np.tile(self.phi, (self.batch_size,self.resp_dur,self.n_loc) ) ) ) - 1.0) ) # stim 2
        Ld             = (self.spon_rate / self.max_delay) * np.ones((self.batch_size,self.max_delay,self.nneuron))                                                   # delay

        R1             = np.random.poisson(L1)
        R2             = np.random.poisson(L2)
        Rd             = np.random.poisson(Ld)

        example_input  = np.concatenate((R1,Rd,R2), axis=1)
        example_output = np.repeat(C[:,np.newaxis],self.total_dur,axis=1)
        example_output = np.repeat(example_output[:,:,np.newaxis],1,axis=2)
        example_mask   = np.ones((self.batch_size,self.total_dur), dtype=np.bool)
        for i in range(self.batch_size):
            example_mask[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay)] = False
            example_input[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0
            example_output[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0        
        
        cum_R1         = np.sum(R1,axis=1) 
        cum_R2         = np.sum(R2,axis=1) 
        
        mu_x           = np.asarray([ np.arctan2( np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ])
        mu_y           = np.asarray([ np.arctan2( np.dot(cum_R2[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R2[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ])
        
        temp_x         = np.asarray([np.swapaxes(np.multiply.outer(cum_R1,cum_R1),1,2)[i,i,:,:] for i in range(self.batch_size)])
        temp_y         = np.asarray([np.swapaxes(np.multiply.outer(cum_R2,cum_R2),1,2)[i,i,:,:] for i in range(self.batch_size)])
        
        kappa_x        = np.asarray( [np.sqrt(np.sum(temp_x[:,i*self.n_in:(i+1)*self.n_in,i*self.n_in:(i+1)*self.n_in] * np.repeat(np.cos( np.subtract(np.expand_dims(self.phi,axis=1), np.expand_dims(self.phi,axis=1).T) )[np.newaxis,:,:],self.batch_size,axis=0),axis=(1,2))) for i in range(self.n_loc) ] )
        kappa_y        = np.asarray( [np.sqrt(np.sum(temp_y[:,i*self.n_in:(i+1)*self.n_in,i*self.n_in:(i+1)*self.n_in] * np.repeat(np.cos( np.subtract(np.expand_dims(self.phi,axis=1), np.expand_dims(self.phi,axis=1).T) )[np.newaxis,:,:],self.batch_size,axis=0),axis=(1,2))) for i in range(self.n_loc) ] )
        
        if self.n_loc==1:
            d          = np.i0(kappa_x) * np.i0(kappa_y) / np.i0( np.sqrt(kappa_x ** 2 + kappa_y ** 2 + 2.0 * kappa_x * kappa_y * np.cos(mu_y-mu_x)) )
        else:
            d          = np.nanmean(np.i0(kappa_x) * np.i0(kappa_y) / np.i0( np.sqrt(kappa_x ** 2 + kappa_y ** 2 + 2.0 * kappa_x * kappa_y * np.cos(mu_y-mu_x)) ), axis=0)
        
        P              = d / (d + 1.0)
        return delay_durs, example_input, example_output, example_mask, C, P


class SineTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_in=1, n_out=1, 
                 stim_dur=25, delay_dur=100, resp_dur=25, alpha=0.0):
        super(SineTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                  
        self.n_out     = n_out
        self.nneuron   = n_in    # total number of input neurons
        self.stim_dur  = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.alpha     = alpha
        self.phi       = np.linspace(0, np.pi, self.resp_dur)
        
    def sample(self):
        
        S              = np.tile(np.sin( self.alpha * self.phi ), (1,1))
        example_input  = np.zeros((self.batch_size, self.total_dur, self.nneuron)) # batch_size x stim_dur x nneuron
        example_input[:,self.stim_dur:(self.stim_dur+self.delay_dur),:] = 0.1 * np.random.randn(self.batch_size, 
                                      self.delay_dur, self.nneuron)
        example_output = np.zeros((self.batch_size, self.total_dur, 1))            # batch_size x stim_dur x 1
        example_output[:,-self.resp_dur:,-1] = S # + 0.01 * np.random.randn(S.shape[0],S.shape[1])
                
        return example_input, example_output, S, S
    

class DelayedEstimationTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_loc=1, n_in=25, n_out=1, stim_dur=10, delay_dur=100, resp_dur=10, kappa=2.0, spon_rate=0.001, tr_cond='all_gains'):
        super(DelayedEstimationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                  # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.spon_rate = spon_rate
        self.nneuron   = self.n_in * self.n_loc # total number of input neurons
        self.phi       = np.linspace(0, np.pi, self.n_in)
        self.stim_dur  = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.tr_cond   = tr_cond
        
    def sample(self):
        
        if self.tr_cond == 'all_gains':
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
                
        S1             = np.pi * np.random.rand(self.n_loc, self.batch_size)
        S              = S1.T
        S1             = np.repeat(S1,self.n_in,axis=0).T
        S1             = np.tile(S1,(self.stim_dur,1,1))
        S1             = np.swapaxes(S1,0,1)
                
        # Noisy responses
        L1             = G * np.exp( self.kappa * (np.cos( 2.0 * (S1 - np.tile(self.phi, (self.batch_size,self.stim_dur,self.n_loc) ) ) ) - 1.0) ) # stim 
        Ld             = (self.spon_rate / self.delay_dur) * np.ones((self.batch_size,self.delay_dur,self.nneuron))                                # delay
        Lr             = (self.spon_rate / self.resp_dur) * np.ones((self.batch_size,self.resp_dur,self.nneuron))                                 # resp

        R1             = np.random.poisson(L1)
        Rd             = np.random.poisson(Ld)
        Rr             = np.random.poisson(Lr)

        example_input  = np.concatenate((R1,Rd,Rr), axis=1)
        example_output = np.repeat(S[:,np.newaxis,:],self.total_dur,axis=1)
        
        cum_R1         = np.sum(R1,axis=1)         
        mu_x           = np.asarray([ np.arctan2( np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ]) / 2.0
        mu_x           = (mu_x > 0.0) * mu_x + (mu_x<0.0) * (mu_x + np.pi) 
        mu_x           = mu_x.T

        return example_input, example_output, S, mu_x


class RajanTask(Task):
    '''Parameters'''

    def __init__(self, max_iter=None, batch_size=50, n_loc=1, n_in=50, n_out=500, stim_dur=10, delay_dur=100, resp_dur=10,
                 kappa=1.0, spon_rate=0.001, tr_cond='all_gains'):
        super(RajanTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in = n_in  # number of neurons per location
        self.n_out = n_out
        self.n_loc = n_loc
        self.kappa = kappa
        self.spon_rate = spon_rate
        self.nneuron = self.n_in * self.n_loc  # total number of input neurons
        self.stim_dur = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.tr_cond = tr_cond
        self.PHI, self.TT = np.meshgrid(np.linspace(0, self.total_dur, self.n_out), np.linspace(0,self.total_dur-1,self.total_dur)) # tile the trial duration

    def sample(self):

        # Noisy responses
        L1 = (10.0 * self.spon_rate / self.stim_dur) * np.ones((self.batch_size, self.stim_dur, self.nneuron))  # dela
        Ld = (self.spon_rate / self.delay_dur) * np.ones((self.batch_size, self.delay_dur, self.nneuron))  # delay
        Lr = (self.spon_rate / self.resp_dur) * np.ones((self.batch_size, self.resp_dur, self.nneuron))  # resp

        R1 = np.random.poisson(L1)
        Rd = np.random.poisson(Ld)
        Rr = np.random.poisson(Lr)
        example_input = np.concatenate((R1, Rd, Rr), axis=1)

        O = np.exp(- (0.5 / self.kappa) * (self.PHI-self.TT)**2 )
        example_output = np.repeat(np.expand_dims(O,axis=0),self.batch_size,axis=0)

        return example_input, example_output, example_output, example_output


class VarDelayedEstimationTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_loc=1, n_in=25, n_out=1, stim_dur=10, max_delay=100, resp_dur=10, kappa=2.0, spon_rate=0.001, tr_cond='all_gains'):
        super(VarDelayedEstimationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                             # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.spon_rate = spon_rate
        self.nneuron   = self.n_in * self.n_loc           # total number of input neurons
        self.phi       = np.linspace(0, np.pi, self.n_in) # np.linspace(0, np.pi, self.n_in)
        self.stim_dur  = stim_dur
        self.max_delay = max_delay
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + max_delay + resp_dur
        self.tr_cond   = tr_cond
        
    def sample(self):
        
        if self.tr_cond == 'all_gains':
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
         
        delay_durs     = np.random.choice([10,40,70,100], size=(self.batch_size,))
        S1             = np.pi * np.random.rand(self.n_loc, self.batch_size)
        S              = S1.T
        S1             = np.repeat(S1,self.n_in,axis=0).T
        S1             = np.tile(S1,(self.stim_dur,1,1))
        S1             = np.swapaxes(S1,0,1)
                
        # Noisy responses
        L1             = G * np.exp( self.kappa * (np.cos( 2.0 * (S1 - np.tile(self.phi, (self.batch_size,self.stim_dur,self.n_loc) ) ) ) - 1.0) ) # stim 
        Ld             = (self.spon_rate / self.max_delay) * np.ones((self.batch_size,self.max_delay,self.nneuron))                                # delay
        Lr             = (self.spon_rate / self.max_delay) * np.ones((self.batch_size,self.resp_dur,self.nneuron))                                 # resp

        R1             = np.random.poisson(L1)
        Rd             = np.random.poisson(Ld)
        Rr             = np.random.poisson(Lr)
        
        example_input  = np.concatenate((R1,Rd,Rr), axis=1)
        example_output = np.repeat(S[:,np.newaxis,:],self.total_dur,axis=1)
        example_mask   = np.ones((self.batch_size,self.total_dur), dtype=np.bool)
        for i in range(self.batch_size):
            example_mask[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay)] = False
            example_input[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0
            example_output[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0        
       
        cum_R1         = np.sum(R1,axis=1)         
        mu_x           = np.asarray([ np.arctan2( np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ]) / 2.0
        mu_x           = (mu_x > 0.0) * mu_x + (mu_x<0.0) * (mu_x + np.pi) 
        mu_x           = mu_x.T
        mu_x           = np.repeat(mu_x[:,np.newaxis,:],self.total_dur,axis=1)
        
        return delay_durs, example_input, example_output, example_mask, S, mu_x

    
class GatedDelayedEstimationTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_loc=2, n_in=25, n_out=1, stim_dur=10, delay_dur=100, resp_dur=10, kappa=2.0, spon_rate=0.001, tr_cond='all_gains'):
        super(GatedDelayedEstimationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                             # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.spon_rate = spon_rate
        self.nneuron   = self.n_in * self.n_loc           # total number of input neurons
        self.phi       = np.linspace(0, np.pi, self.n_in) # np.linspace(0, np.pi, self.n_in)
        self.stim_dur  = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.tr_cond   = tr_cond
        
    def sample(self):
        
        if self.tr_cond == 'all_gains':
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)

        C              = np.random.choice([0.0, 1.0], size=(self.batch_size,))
        C0ind          = np.where(C==0.0)[0]        # change
        C1ind          = np.where(C==1.0)[0]        # change
                
        S1             = np.pi * np.random.rand(self.n_loc, self.batch_size)
        Sboth          = S1.T
        S              = np.expand_dims(Sboth[:,0],axis=1) 
        S[C1ind,0]     = Sboth[C1ind,1] 

        S1             = np.repeat(S1,self.n_in,axis=0).T
        S1             = np.tile(S1,(self.stim_dur,1,1))
        S1             = np.swapaxes(S1,0,1)
                
        # Noisy responses
        L1 = G * np.exp( self.kappa * (np.cos( 2.0 * (S1 - np.tile(self.phi, (self.batch_size,self.stim_dur,self.n_loc) ) ) ) - 1.0) )  # stim
        Ld = (self.spon_rate / self.delay_dur) * np.ones((self.batch_size,self.delay_dur,self.nneuron))     # delay
        Lr = (self.spon_rate / self.resp_dur) * np.ones((self.batch_size,self.resp_dur,self.nneuron))
        Lr[C0ind,:,:self.n_in] = 5.0*Lr[C0ind,:,:self.n_in]                                                 # cue 0
        Lr[C1ind,:,self.n_in:] = 5.0*Lr[C1ind,:,self.n_in:]                                                # cue 1

        R1             = np.random.poisson(L1)
        Rd             = np.random.poisson(Ld)
        Rr             = np.random.poisson(Lr)

        example_input  = np.concatenate((R1,Rd,Rr), axis=1)
        example_output = np.repeat(S[:,np.newaxis,:],self.total_dur,axis=1)
        
        cum_R1         = np.sum(R1,axis=1)         
        mu_x           = np.asarray([ np.arctan2( np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ]) / 2.0
        mu_x           = (mu_x > 0.0) * mu_x + (mu_x<0.0) * (mu_x + np.pi) 
        mu_x           = mu_x.T
        mu_aux         = np.expand_dims(mu_x[:,0],axis=1)
        mu_aux[C1ind,0] = mu_x[C1ind,1]
        
        return example_input, example_output, S, mu_aux


class VarGatedDelayedEstimationTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=50, n_loc=2, n_in=25, n_out=1, stim_dur=10, max_delay=100, resp_dur=10, kappa=2.0, spon_rate=0.001, tr_cond='all_gains'):
        super(VarGatedDelayedEstimationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in                             # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.spon_rate = spon_rate
        self.nneuron   = self.n_in * self.n_loc           # total number of input neurons
        self.phi       = np.linspace(0, np.pi, self.n_in) # np.linspace(0, np.pi, self.n_in)
        self.stim_dur  = stim_dur
        self.max_delay = max_delay
        self.resp_dur = resp_dur
        self.total_dur = stim_dur + max_delay + resp_dur
        self.tr_cond   = tr_cond
        
    def sample(self):
        
        if self.tr_cond == 'all_gains':
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)

        C              = np.random.choice([0.0, 1.0], size=(self.batch_size,))
        C0ind          = np.where(C==0.0)[0]        # change
        C1ind          = np.where(C==1.0)[0]        # change

        delay_durs     = np.random.choice([10,40,70,100], size=(self.batch_size,))                
        S1             = np.pi * np.random.rand(self.n_loc, self.batch_size)
        Sboth          = S1.T
        S              = np.expand_dims(Sboth[:,0],axis=1) 
        S[C1ind,0]     = Sboth[C1ind,1] 

        S1             = np.repeat(S1,self.n_in,axis=0).T
        S1             = np.tile(S1,(self.stim_dur,1,1))
        S1             = np.swapaxes(S1,0,1)
                
        # Noisy responses
        L1             = G * np.exp( self.kappa * (np.cos( 2.0 * (S1 - np.tile(self.phi, (self.batch_size,self.stim_dur,self.n_loc) ) ) ) - 1.0) ) # stim 
        Ld             = (self.spon_rate / self.max_delay) * np.ones((self.batch_size,self.max_delay,self.nneuron))                                                   # delay
        Lr             = (self.spon_rate / self.resp_dur) * np.ones((self.batch_size,self.resp_dur,self.nneuron))    
        Lr[C0ind,:,:self.n_in] = 5.0*Lr[C0ind,:,:self.n_in]                                                 # cue 0
        Lr[C1ind,:,self.n_in:] = 5.0*Lr[C1ind,:,self.n_in:]                                               # cue 1

        R1             = np.random.poisson(L1)
        Rd             = np.random.poisson(Ld)
        Rr             = np.random.poisson(Lr)

        example_input  = np.concatenate((R1,Rd,Rr), axis=1)
        example_output = np.repeat(S[:,np.newaxis,:],self.total_dur,axis=1)
        example_mask   = np.ones((self.batch_size,self.total_dur), dtype=np.bool)
        for i in range(self.batch_size):
            example_mask[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay)] = False
            example_input[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0
            example_output[i,(self.stim_dur+delay_durs[i]):(self.stim_dur+self.max_delay),:] = 0.0        
        
        cum_R1         = np.sum(R1,axis=1)         
        mu_x           = np.asarray([ np.arctan2( np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ]) / 2.0
        mu_x           = (mu_x > 0.0) * mu_x + (mu_x<0.0) * (mu_x + np.pi) 
        mu_x           = mu_x.T
        mu_aux         = np.expand_dims(mu_x[:,0],axis=1)
        mu_aux[C1ind,0] = mu_x[C1ind,1]
        
        return delay_durs, example_input, example_output, example_mask, S, mu_aux