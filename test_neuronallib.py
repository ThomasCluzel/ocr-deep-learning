# -*-coding:utf-8-*
"""
This file contains all the tests relative to
the neuronal library.
"""

import os
import pytest
import shutil

from neuronallib import NeuronalNetwork

# At first test the static methods
def test_get_char_from_index_list():
    assert NeuronalNetwork.get_char_from_index_list([0,1,2,9], "OCR_data", "ocr") == \
        ['0', '1', '2', '9']

def test_load_data():
    data = NeuronalNetwork.load_data("OCR_data", "ocr", 63)
    assert data is not None
    assert data.train is not None
    assert data.train.images is not None
    assert data.train.labels is not None
    assert len(data.train.images) == len(data.train.labels)
    assert data.test is not None
    assert data.test.images is not None
    assert data.test.labels is not None
    assert len(data.test.images) == len(data.test.labels)

# After test the class
class TestNeuronalNetwork:
    """
    This class has been design to test the behaviour
    of a neuronal network.
    """
    def setup_class(self):
        """
        Initialize variables
        """
        self.tmp_cnn_dir_name = "tmp"
        os.mkdir(self.tmp_cnn_dir_name)
        self.data = NeuronalNetwork.load_data("OCR_data", "ocr", 63)

    def teardown_class(self):
        """
        Clear variables
        """
        shutil.rmtree(self.tmp_cnn_dir_name)

    def test_network_creation(self):
        nn = NeuronalNetwork(0.001, nb_output_classes=63, \
            save_filename=os.path.join(self.tmp_cnn_dir_name, "model")) # high learning rate, fast training
        assert nn is not None

    def test_network_training(self):
        # assuming the creation is not bugged
        nn = NeuronalNetwork(0.001, nb_output_classes=63, \
            save_filename=os.path.join(self.tmp_cnn_dir_name, "model"))
        with pytest.raises(RuntimeError):
            nn.train(None) # No data raises a RuntimeError
        nn.train(self.data, 20)
        # can be trained multiple time
        nn.train(self.data, 10)
    
    def test_network_loading(self):
        nn = NeuronalNetwork(os.path.join(self.tmp_cnn_dir_name, "model"))
        assert nn is not None

    # TODO : really train a network before using this function
    def test_network_usage(self):
        # assuming loading is not bugged
        nn = NeuronalNetwork(os.path.join(self.tmp_cnn_dir_name, "model"))
        with pytest.raises(ValueError):
            nn.guess(2) # Wrong type will raise a ValueError
        nb_test = 20
        picts = [ self.data.test.images[i] for i in range(nb_test) ]
        correct = NeuronalNetwork.get_char_from_index_list([ self.data.test.labels[i].tolist().index(1) for i in range(nb_test) ])
        numbers, prob = nn.guess(picts)
        rate = sum(list(map(lambda x, y: x==y and 1 or 0, numbers, correct))) / nb_test
        print("Correct rate (%d tries) is: %f" % (nb_test, rate), "\n  Nums  ==>", numbers, "\n  Prob  ==>", prob.round(2))
        # TODO : add an assertion with the rate ?
