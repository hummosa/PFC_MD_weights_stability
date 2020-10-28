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

