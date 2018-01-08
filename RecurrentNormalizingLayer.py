import lasagne
import theano


class LayerNormalization(object):

    def __init__(self, num_units,
                 nonlinearity=lasagne.nonlinearities.softplus,
                 b_norm=lasagne.init.Constant(0.0),
                 g_norm=lasagne.init.Constant(1.0),
                 eps_norm=0.0001):
        self.num_units = num_units
        self.b_norm = theano.shared(b_norm.sample(num_units), name='layer_norm.b_norm')
        #self.g_norm = theano.shared(g_norm.sample(num_units), name='layer_norm.g_norm')
        self.eps_norm = eps_norm
        self.nonlinearity = nonlinearity

    def normalizing_nonlinearity(self, x):
        m = x.mean(-1, keepdims=True) + self.eps_norm
        #s = x.std(-1, keepdims=True) + self.eps_norm
        b_norm = self.b_norm.reshape((1,) * (x.ndim - 1) + (-1,))
        #g_norm = self.g_norm.reshape((1,) * (x.ndim - 1) + (-1,))
        #return self.nonlinearity(g_norm * x + b_norm)
        return self.nonlinearity(1.0 * x / m + b_norm)

    def register_to(self, layer):
        #pass
        layer.add_param(self.b_norm, (self.num_units,))
        #layer.add_param(self.g_norm, (self.num_units,))


class RecurrentNormalizingLayer(lasagne.layers.CustomRecurrentLayer):

    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 num_units,
                 **kwargs):
        self.layer_normalization = LayerNormalization(num_units)
        super(RecurrentNormalizingLayer, self).__init__(
            incoming, input_to_hidden, hidden_to_hidden,
            nonlinearity=self.layer_normalization.normalizing_nonlinearity,
            **kwargs)
        self.layer_normalization.register_to(self.hidden_to_hidden)
