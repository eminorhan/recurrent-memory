# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 20:00:31 2016 by emin
"""

from lasagne.init import Initializer
from lasagne.utils import floatX
import numpy as np

class LeInit(Initializer):
    """Initialize weights with diagonal matrix.
    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, diag_val=1.0, offdiag_val=0.0):
        self.diag_val = diag_val
        self.offdiag_val = offdiag_val

    def sample(self, shape):
        my_off_diag = self.offdiag_val*np.random.randn(shape[0],shape[0])
        return floatX(np.eye(shape[0]) * self.diag_val + my_off_diag - np.diag(np.diag(my_off_diag)))