# classes for training, testing, experimenting or running any simulations ...

class Train_model():
    """
    Class to hold model training procedure parameters and algorithm..
    """
    def __init__(self, parameter_list):
        """
        Initialize class with default parametes
        """

        raise NotImplementedError

    def train(self, model, Ntrain):
        """
        trains model model for Ntrain iterations
        """
        for traini in range(Ntrain):
            pass

        return model #return trained model


class Test_model():
    """
    Test the performance of a model, also plots exemplar neuronal responses for a given list of input combinations
    """

    def __init__(self, parameter_list):
        """
        Initialize class with default parametes
        """
        raise NotImplementedError

    def test(self, model, Ntest):
        """
        tests model model for Ntest iterations
        """
        for testi in range(Ntest):
            pass

        
class Experiment():
    def __init__(self, model, parameter_list):
        trainer = Train_model(parameter_list)
        tester  = Test_model(parameter_list)

        trainer.train(model, 1000)
        #plot training weights, average firing rates.
        tester.test(model, 1000)
        #plot results, performance, exemplar neuron responses for each area.
