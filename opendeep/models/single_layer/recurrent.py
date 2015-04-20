"""
.. module:: recurrent

This module provides different recurrent networks
"""
__authors__ = "Markus Beissinger, Skylar Payne"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# internal references
from opendeep.models.model import Model
from opendeep.models.single_layer.basic import BasicLayer
from opendeep.utils.nnet import get_weights_gaussian, get_weights_uniform, get_bias
from opendeep.utils.activation import get_activation_function
from opendeep.utils.cost import get_cost_function
#theano
import theano
import theano.tensor as T
#external
import numpy as np

log = logging.getLogger(__name__)

class RecurrentLayer(Model):
    """
    Your run-of-the-mill recurrent model. Normally not as good as LSTM/GRU, but it is simplest.
    """
    default = {
        'activation': 'tanh', #type of activation function to use for output
        'cost': 'mse',
        'cost_args': {},
        'weights_init': 'gaussian', #either 'gaussian' or 'uniform' - how to initialize weights
        'weights_mean': 0, #mean for gaussian weights init
        'weights_std': 0.005, #standard deviation for gaussian weights init
        'weights_interval': 'montreal', # if the weights_init was 'uniform' how to initialize from uniform
        'bias_init': 0.0, #how to initialize the bias parameter
        'input_size': None,
        'output_size': None
    }
    #TODO: [output, input1, input2, input3] -> [0, -3, -2, -1]. Is this interface confusing?
    def __init__(self, inputs_hook=None, config=None, defaults=default, params_hook=None,
                input_size=None, output_size=None, activation=None, weights_init=None,
                weights_mean=None, weights_std=None, weights_interval=None, bias_init=None,
                transition=None, taps={0: [-1], -1: [0]}, go_backwards=False, **kwargs):
        #init Model to combine defaults and config dictionaries.
        super(RecurrentLayer, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})
        # all configuration parameters are now in self.args

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, or from parameters
        if inputs_hook is not None: # inputs_hook is a tuple of (Shape, Input)
            assert len(inputs_hook) == 2 # make sure inputs_hook is a tuple
            self.input_size = inputs_hook[0] or self.input_size
            self.input = self.inputs_hook[1]
        else:
            # make the iput a symbolic matrix
            #TODO: split this line
            self.input = T.fmatrix('X') if type(self.input_size) is not list else [T.fmatrix('X%d' % i) for i in xrange(len(self.input_size))]
       
        #RNN can take in inputs from multiple sources. Store in a list:
        if type(self.input) is not list:
            self.input = [self.input]
            self.input_size = [self.input_size]

        #now that we have input specs, define output 'target' variable for supervised training
        self.target = T.fmatrix('Y')

        # either grab from the paramter directly, self.args config, or copy n_in
        self.output_size = self.output_size or self.input_size
        
        self.taps = taps
        if 0 not in self.taps:
            self.taps[0] = []
        
        #cost function
        #if a string name was given, look up the correct function
        if isinstance(self.cost, basestring):
            cost_func = get_cost_function(self.cost)
        #otherwise, if a callable (custom function) was passed:
        else:
            assert callable(self.cost), "Cost function either needs to be a string name or callable!"
            cost_func = self.cost


        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        #you can either specify a transition function, or a basic layer will be used by default.
        if transition is not None:
            #TODO: assert that it is of type model -- assert input and output size?
            raise NotImplementedError("Deep Transitions for RNN is not yet implemented")
        else:
            #need to find the input_size for the basic layer
            # this is essentially sum of the # taps * input_size
            # but we don't know anything about any other layers from here.
            # how much work do we shift to the user?
            #for an initial go, let's just use size of input and size of output
            all_sizes = [self.output_size] + self.input_size
            transition_input_size = sum([all_sizes[k] * len(v) for k,v in self.taps.iteritems()]) 
            
            transition_args = {arg: val for (arg, val) in locals().iteritems() if arg is not 'self'}
            transition_args['inputs_hook'] = (transition_input_size, T.fvector('X_transition'))
            self.transition = BasicLayer(**transition_args)
        #Nothing about the params_hook because that's taken care of in transition

        ###############
        # computation #
        ###############
        #TODO: fix formatting
        self.output, _ = theano.scan(self._step,
                    sequences=[dict(input=self.input[k], taps=v) for k,v in self.taps.iteritems() if k is not 0], #first dimension for all inputs should be Time
                    outputs_info=[dict(initial=T.zeros_like(self.transition.output), taps=self.taps[0])], #The output at the time step before your first input
                    go_backwards=go_backwards)

        self.cost = cost_func(output=self.output, target=self.target, **self.cost_args) 
        log.debug("Initialized a recurrent, fully-connected layer with shape %s and transition: %s" %
                    (str((self.input_size, self.output_size)), str(self.transition)))
    
    def _step(self, *args):
        for i,a in enumerate(args):
            log.info("X%d - %s" % (i, str(a.type)))
        full_input = T.concatenate([a.flatten() for a in args],axis=0)
        replaces = {self.transition.input: full_input}
        out = theano.clone(self.transition.output, replace=replaces)
        return out

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_targets(self):
        return [self.target]
    
    def get_train_cost(self):
        return self.cost

    def get_params(self):
        return self.params

    def save_args(self, args_file="recurrentlayer_config.pkl"):
        super(RecurrentLayer, self).save_args(args_file)

class LSTM(Model):
    """
    Long short-term memory units.
    https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/lstm.py
    """
    # TODO: LSTM recurrent model
    def __init__(self):
        log.error("LSTM not implemented!")
        raise NotImplementedError("LSTM not implemented!")

class GRU(Model):
    """
    Gated recurrent units.
    """
    # TODO: GRU recurrent model
    def __init__(self):
        log.error("GRU not implemented!")
        raise NotImplementedError("GRU not implemented!")
