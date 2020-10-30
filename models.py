# classes for PFC generic class, MD class. and an all encompassing Model class
'''
    Does input need to be a class? 
    How about output neurons?
'''


class Layer_model():
    """
    Class to hold a generic PFC model
    """

    def __init__(self, parameter_list):
        """
        Initialize Model with default parametes
        """

        raise NotImplementedError

    def step(self, inputs):
        '''
        takes a list of inputs
        outputs a list of outputs
        '''
        outputs = 2 * inputs

        return outputs


class PFC(Layer_model):
    '''

    '''
    pass


class MD(Layer_model):
    '''
    '''
    pass


class Model():
    """
    Defines a number of areas or layers
    Holds the weights connecting them
    Has a step function that steps through layers one by one and gives outputs
    possibly attach a weight_update or a train function (vs. have that separately handled by the simulations.py/class train_model ?)
    """

    def __init__(self, parameter_list):
        """
        Initialize class with default parametes
        """
        self.RNNs = []
        self.pfc = PFC()
        self.ofc = PFC()
        self.md = MD()

        self.RNNs.append(self.pfc)
        self.RNNs.append(self.ofc)

        raise NotImplementedError

    def step(self, inputs):
        """
        Steps through all models
        """
        md_outputs = self.md(inputs, pfc_outputs, ofc_outputs)
        pfc_outputs = self.pfc(inputs, ofc_outputs, md_outputs)
        # THIS is obviously circular!!! some init values at zero?  attach current state of each layer output to the Model class and update them each step, init to zeros
        ofc_outputs = self.ofc(inputs, pfc_outputs, md_outputs)

        return pfc_outputs
