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
from opendeep.utils.nnet import get_weights_gaussian, get_weights_uniform, get_bias
from opendeep.utils.activation import get_activation_function
#theano
import theano
import theano.tensor as T

log = logging.getLogger(__name__)

class RecurrentLayer(Model):
    """
    Your run-of-the-mill recurrent model. Normally not as good as LSTM/GRU, but it is simplest.
    """
    default = {
        'activation': 'tanh', #type of activation function to use for output
        'weights_init': 'gaussian', #either 'gaussian' or 'uniform' - how to initialize weights
        'weights_mean': 0, #mean for gaussian weights init
        'weights_std': 0.005, #standard deviation for gaussian weights init
        'weights_interval': 'montreal', # if the weights_init was 'uniform' how to initialize from uniform
        'bias_init': 0.0, #how to initialize the bias parameter
        'output_t0_init': 0.0
    }

    def __init__(self, inputs_hook=None, config=None, defaults=default, params_hook=None,
                input_size=None, output_size=None, activation=None, weights_init=None,
                weights_mean=None, weights_std=None, weights_interval=None, bias_init=None, output_t0_init=None):
        #init Model to combine defaults and config dictionaries.
        super(RecurrentLayer, self).__init__(config, defaults)
        # all configuration parameters are now in self.args

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, or from parameters
        if inputs_hook: # inputs_hook is a tuple of (Shape, Input)
            assert len(inputs_hook) == 2 # make sure inputs_hook is a tuple
            input_size = inputs_hook[0] or input_size
            self.input = inputs_hook[1]
        else:
            #either grab from the parameter directly or self.args config
            input_size = input_size or self.args.get('input_size')
            # make the iput a symbolic matrix
            self.input = T.fmatrix('X')
        # either grab from the paramter directly, self.args config, or copy n_in
        output_size = output_size or self.args.get('output_size') or input_size

        # other specifications
        weights_init = weights_init or self.args.get('weights_init')
        #for gaussian weights
        mean = weights_mean or self.args.get('weights_mean')
        std = weights_std or self.args.get('weights_std')
        # for uniform weights
        interval = weights_interval or self.args.get('weights_interval')
        # for bias
        bias_init = bias_init or self.args.get('bias_init')
        # for initial output setting
        output_t0_init = output_t0_init or self.args.get('output_t0_init')

        #activation function
        activation_name = activation or self.args.get('activation')
        if isinstance(activation_name, basestring):
            activation_func = get_activation_function(activation_name)
        else:
            assert callable(activation_name)
            activation_func = activation_name

        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if params_hook:
            assert len(params_hook) == 4, "Expected 4 params (Wx, W, output_t0, and b) for RecurrentLayer, found {0!s}".format(
                    len(params_hook)) # make sure the params_hook has Wx, W, and b
            Wx, W, output_t0, b = params_hook
        else:
            # if we are initializing weights from a gaussian
            if weights_init.lower() == 'gaussian':
                Wx = get_weights_gaussian(shape=(input_size, output_size), mean=mean, std=std, name="Wx")
                W = get_weights_gaussian(shape=(output_size, output_size), mean=mean, std=std, name="W")
            # if we are initializing weights from a uniform distribution
            elif weights_init.lower() == 'uniform':
                Wx = get_weights_uniform(shape=(input_size, output_size), mean=mean, std=std, name="Wx")
                W = get_weights_uniform(shape=(output_size, output_size), mean=mean, std=std, name="W")
            # otherwise not implemented
            else:
                log.error("Did not recognize weights_init %s! Please try gaussian or uniform" % 
                            str(weights_init))
                raise NotImplementedError("Did not recognize weights_init %s! Please try gaussian or uniform" %
                            str(weights_init))

            b = get_bias(shape=output_size, name="b", init_values=bias_init)
            #NOTE: Since output_t0 is usually initialized similar to biases in the literature,
            # I used get_bias for initialization
            output_t0 = get_bias(shape=output_size, name="output_t0", init_values=output_t0_init)
        # Finally have the three parameters!
        self.params = [Wx, W, output_t0, b]

        ###############
        # computation #
        ###############
        #This defines a single recurrent step where x_t is the current input, and o_tm1 is the output from
        # the previous step.
        def step(x_t, o_tm1):
            return T.tanh(T.dot(x_t, Wx) + T.dot(o_tm1, W) + b)

        output, _ = theano.scan(step,
                    sequences=self.input, #First dimension of self.input should be time.
                    outputs_info=output_t0) #The output at the time step before your first input
        
        self.output = output
        log.debug("Initialized a recurrent, fully-connected layer with shape %s and activation: %s" %
                    (str((input_size, output_size)), str(activation_name)))

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_params(self):
        return self.params

class LSTM(Model):
    """
    Long short-term memory units.
    https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/lstm.py
    """
    # TODO: LSTM recurrent model
    log.error("LSTM not implemented!")
    raise NotImplementedError("LSTM not implemented!")

class GRU(Model):
    """
    Gated recurrent units.
    """
    # TODO: GRU recurrent model
    log.error("GRU not implemented!")
    raise NotImplementedError("GRU not implemented!")
