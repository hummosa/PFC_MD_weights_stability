import os
import pickle


def save_network(filename, net):
    if not os.path.exists('data'):
        os.makedirs('data')

    filehandler = open(f'data/{filename}', 'wb')
    pickle.dump(net, filehandler)
    filehandler.close()


def load_network(filename):
    filehandler = open(f'data/{filename}', 'rb')
    data = pickle.load(filehandler)
    filehandler.close()
    return data
