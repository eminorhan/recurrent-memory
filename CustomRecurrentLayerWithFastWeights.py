#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:44:56 2017 @author: emin
"""
import numpy as np
import theano
import theano.tensor as T
from lasagne import init, nonlinearities
from lasagne.utils import unroll_scan
from lasagne.layers import Layer, InputLayer, MergeLayer
from lasagne.layers import helper

class CustomRecurrentLayerWithFastWeights(MergeLayer):
    """ """
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 gamma=0.9,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        super(CustomRecurrentLayerWithFastWeights, self).__init__(incomings, **kwargs)

        input_to_hidden_in_layers = \
            [layer for layer in helper.get_all_layers(input_to_hidden)
             if isinstance(layer, InputLayer)]
        if len(input_to_hidden_in_layers) != 1:
            raise ValueError(
                '`input_to_hidden` must have exactly one InputLayer, but it '
                'has {}'.format(len(input_to_hidden_in_layers)))

        hidden_to_hidden_in_lyrs = \
            [layer for layer in helper.get_all_layers(hidden_to_hidden)
             if isinstance(layer, InputLayer)]
        if len(hidden_to_hidden_in_lyrs) != 1:
            raise ValueError(
                '`hidden_to_hidden` must have exactly one InputLayer, but it '
                'has {}'.format(len(hidden_to_hidden_in_lyrs)))
        hidden_to_hidden_in_layer = hidden_to_hidden_in_lyrs[0]

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.gamma = gamma

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Check that the input_to_hidden connection can appropriately handle
        # a first dimension of input_shape[0]*input_shape[1] when we will
        # precompute the input dot product
        if (self.precompute_input and
                input_to_hidden.output_shape[0] is not None and
                input_shape[0] is not None and
                input_shape[1] is not None and
                (input_to_hidden.output_shape[0] !=
                 input_shape[0]*input_shape[1])):
            raise ValueError(
                'When precompute_input == True, '
                'input_to_hidden.output_shape[0] must equal '
                'incoming.output_shape[0]*incoming.output_shape[1] '
                '(i.e. batch_size*sequence_length) or be None but '
                'input_to_hidden.output_shape[0] = {} and '
                'incoming.output_shape[0]*incoming.output_shape[1] = '
                '{}'.format(input_to_hidden.output_shape[0],
                            input_shape[0]*input_shape[1]))

        # Check that the first dimension of input_to_hidden and
        # hidden_to_hidden's outputs match when we won't precompute the input
        # dot product
        if (not self.precompute_input and
                input_to_hidden.output_shape[0] is not None and
                hidden_to_hidden.output_shape[0] is not None and
                (input_to_hidden.output_shape[0] !=
                 hidden_to_hidden.output_shape[0])):
            raise ValueError(
                'When precompute_input == False, '
                'input_to_hidden.output_shape[0] must equal '
                'hidden_to_hidden.output_shape[0] but '
                'input_to_hidden.output_shape[0] = {} and '
                'hidden_to_hidden.output_shape[0] = {}'.format(
                    input_to_hidden.output_shape[0],
                    hidden_to_hidden.output_shape[0]))

        # Check that input_to_hidden and hidden_to_hidden output shapes match,
        # but don't check a dimension if it's None for either shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[1:],
                                     hidden_to_hidden.output_shape[1:])):
            raise ValueError("The output shape for input_to_hidden and "
                             "hidden_to_hidden must be equal after the first "
                             "dimension, but input_to_hidden.output_shape={} "
                             "and hidden_to_hidden.output_shape={}".format(
                                 input_to_hidden.output_shape,
                                 hidden_to_hidden.output_shape))

        # Check that input_to_hidden's output shape is the same as
        # hidden_to_hidden's input shape but don't check a dimension if it's
        # None for either shape
        h_to_h_input_shape = hidden_to_hidden_in_layer.output_shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[1:],
                                     h_to_h_input_shape[1:])):
            raise ValueError(
                "The output shape for input_to_hidden must be equal to the "
                "input shape of hidden_to_hidden after the first dimension, "
                "but input_to_hidden.output_shape={} and "
                "hidden_to_hidden:input_layer.shape={}".format(
                    input_to_hidden.output_shape, h_to_h_input_shape))

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1,) + hidden_to_hidden.output_shape[1:],
                name="hid_init", trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(CustomRecurrentLayerWithFastWeights, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        return params

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return (input_shape[0],) + self.hidden_to_hidden.output_shape[1:]
        # Otherwise, the shape will be (n_batch, n_steps, trailing_dims...)
        else:
            return ((input_shape[0], input_shape[1]) +
                    self.hidden_to_hidden.output_shape[1:])

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]

        if self.precompute_input:
            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, batch_size, trailing dimensions...) to
            # (seq_len*batch_size, trailing dimensions...)
            # This strange use of a generator in a tuple was because
            # input.shape[2:] was raising a Theano error
            trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
            input = T.reshape(input, (seq_len*num_batch,) + trailing_dims)
            input = helper.get_output(self.input_to_hidden, input, **kwargs)

            # Reshape back to (seq_len, batch_size, trailing dimensions...)
            trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
            input = T.reshape(input, (seq_len, num_batch) + trailing_dims)

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        # When we are not precomputing the input, we also need to pass the
        # input-to-hidden parameters to step
        if not self.precompute_input:
            non_seqs += helper.get_all_params(self.input_to_hidden)

        # Create single recurrent computation step function
        def step(input_n, hid_prevprev, hid_previous, *args):
            # Compute the hidden-to-hidden activation
            hid_pre = helper.get_output(self.hidden_to_hidden, hid_previous, **kwargs)

            # If the dot product is precomputed then add it, otherwise
            # calculate the input_to_hidden values and add them
            if self.precompute_input:
                hid_pre += input_n
            else:
                hid_pre += helper.get_output(
                    self.input_to_hidden, input_n, **kwargs)

            # Clip gradients
            if self.grad_clipping:
                hid_pre = theano.gradient.grad_clip(hid_pre, -self.grad_clipping, self.grad_clipping)

            hid_pre += self.gamma * hid_prevprev * T.clip(T.tile(T.reshape(T.diagonal(T.dot(hid_prevprev, hid_previous.T)),
                                                      (1,hid_previous.shape[0])), (hid_previous.shape[1],1)).T, 0.0, 100.0)

            return self.nonlinearity( hid_pre )


        def step_masked(input_n, mask_n, hid_previous, *args):
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = step(input_n, hid_previous, *args)
            hid_out = T.switch(mask_n, hid, hid_previous)
            return [hid_out]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # The code below simply repeats self.hid_init num_batch times in
            # its first dimension.  Turns out using a dot product and a
            # dimshuffle is faster than T.repeat.
            dot_dims = (list(range(1, self.hid_init.ndim - 1)) +
                        [0, self.hid_init.ndim - 1])
            hid_init = T.dot(T.ones((num_batch, 1)),
                             self.hid_init.dimshuffle(dot_dims))

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[dict(initial=T.zeros((2,num_batch,500)),taps=[-2,-1])],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[dict(initial=T.zeros((2,num_batch,500)),taps=[-2,-1])],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, *range(2, hid_out.ndim))

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out    