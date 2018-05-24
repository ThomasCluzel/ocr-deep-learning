# -*-coding:utf-8-*
"""
The neuronal library
====================

This file contains the neuronal network.
It can be initialized, trained, saved and used
to guess characters written on 28x28 pixels pictures.

Use (example):
from neuronallib import NeuronalNetwork
nn = NeuronalNetwork(NeuronalNetwork.SAVE_FILE)
characters, probabilities = nn.guess(pictures)
"""


# libraries
import os
import time
import numpy as np
import tensorflow as tf

from datamanager import read_data_sets


class NeuronalNetwork:
    """
    This class manages the neuronal network.
    It can create (from scratch) a neuronal netwok, load an already
    existing network, use a network to guess a character, train a network
    and save a network.

    Attributes:
        - self._graph: tf.Graph, the network itself
        - self._saver: tf.train.Saver, the object to store the graph in files
        - self._save_filename: str, the name of the file (without extension) within the graph is saved
        - self._nb_output_classes: int, th number of output classes (number of elements recognisable)
    """

    # Class attributes (default values) and methodes
    SAVE_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("cnn", "model"))
    NB_OUTPUT_CLASSES = 63           # for Optical Character Recognition
    DATABASE_DIR = "OCR_data_v2"
    DATABASE_NAME = "ocr"
    # NB_OUTPUT_CLASSES = 10          # for handwritten digits
    # DATABASE_DIR = "EMNIST_data"
    # DATABASE_NAME = "emnist-mnist"

    @staticmethod
    def get_char_from_index_list(index_list, dirname=DATABASE_DIR, filename=DATABASE_NAME):
        """
        Returns the list of character corresponding to the index
        in the index list compute by the neuronal network.

        :param index_list: List return by the evaluation of the
            output node of the graph.
        :param dirname: a str, the directory containing the database files.
        :param filename: a str, the begining of the names of all files of the database.
        :return: A list of characters "read" by the network.
        """
        with open(os.path.join(dirname, filename) + "-mapping.txt", 'r') as f:
            couples = [ ligne.split(' ') for ligne in f.read().split('\n') ]
        couples = [ (int(a), int(b)) for a,b in couples ]
        return [ chr([ b for a,b in couples if a==index][0]) for index in index_list ]

    @staticmethod
    def load_data(dirname=DATABASE_DIR, filename=DATABASE_NAME, nb_classes=NB_OUTPUT_CLASSES):
        """
        Returns a DataSet object loading from a database in MNIST format.

        :param dirname: a str, the directory containing the database files.
        :param filename: a str, the begining of the names of all files of the database.
        :param nb_classes: an integer, the number of output classes
        :return: a DateSet
        """
        return read_data_sets(train_dir = dirname,
                              train_images = filename+'-train-images-idx3-ubyte.gz',
                              train_labels = filename+'-train-labels-idx1-ubyte.gz',
                              test_images = filename+'-test-images-idx3-ubyte.gz',
                              test_labels = filename+'-test-labels-idx1-ubyte.gz',
                              num_classes = nb_classes,
                              one_hot=True)

    def __init__(self, learning_rate_or_save_filename=0.0001, hard_examples=False,
            nb_output_classes=NB_OUTPUT_CLASSES, save_filename=SAVE_FILE):
        """
        Load the network saved in the given file if any
        or generate a network to train with given parameters.

        First case: Create a network from scratch
            nn = NeuronalNetwork() # with default parameters
            nn = NeuronalNetwork(0.001, True) # learning rate = 0.001 and hard examples activated
        Second case: Load a network
            nn = NeuronalNetwork(NeuronalNetwork.SAVE_FILE) # load a graph stored in the default directory

        :param learning_rate_or_save_filename: In the first case: a float, the learning rate.
            In the second case: a str, the name of the file containing the saved network to load.
        :param hard_examples: only in the first case: a boolean, True if the network
            shall use the hard examples learning method
        :param save_filename: only in the first case: a str, the path to save the network.
        """
        # create or load ?
        if(not isinstance(learning_rate_or_save_filename, str)):
            # Create the network
            self._nb_output_classes = nb_output_classes
            self._save_filename = save_filename
            self._graph = tf.Graph()
            self._create_network()
            self._set_training_params(learning_rate_or_save_filename, hard_examples)
        else:
            # load the saved network
            self._save_filename = learning_rate_or_save_filename
            self._load()
            self._graph = tf.get_default_graph()

    def _create_network(self):
        """
        Create the network from scratch.
        """
        # Functions useful to create the graph
        def weight_variable(shape):
            """Generate a non-void matrix"""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            """Generate a non-void vector"""
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        # Graph initialization
        self._graph.as_default()
        # inputs of the graph
        with tf.variable_scope("Inputs", reuse=tf.AUTO_REUSE):
            x = tf.placeholder(tf.float32, [None, 784]) # 28x28 = 784
            y_ = tf.placeholder(tf.float32, [None, self._nb_output_classes])
        # the first hidden/convolutional layer
        with tf.variable_scope("First_layer", reuse=tf.AUTO_REUSE):
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
        # the second hidden/convolutional layer
        with tf.variable_scope("Second_layer", reuse=tf.AUTO_REUSE):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
        # last hidden layer
        with tf.variable_scope("Third_layer", reuse=tf.AUTO_REUSE):
            W_fc1 = weight_variable([7*7*64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropout to avoid overfitting
        with tf.variable_scope("Dropout", reuse=tf.AUTO_REUSE):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # last layer
        with tf.variable_scope("Last_layer", reuse=tf.AUTO_REUSE):
            W_fc2 = weight_variable([1024, self._nb_output_classes])
            b_fc2 = bias_variable([self._nb_output_classes])
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            y_output = tf.argmax(y_conv, 1)
            probability_output = tf.reduce_max(tf.nn.softmax(y_conv), axis=[1])
        # add placeholders and output to the collection to have access to
        # them later when guess numbers
        graph = tf.get_default_graph()
        graph.add_to_collection("x", x)
        graph.add_to_collection("y_", y_)
        graph.add_to_collection("keep_prob", keep_prob)
        graph.add_to_collection("y_conv", y_conv)
        graph.add_to_collection("y_output", y_output)
        graph.add_to_collection("probability_output", probability_output)

    def _load(self):
        """
        Load a network saved in the file self._save_filename.

        :param save_filename: The name of the file containing the saved
            network to load.
        :type save_filename: str
        """
        self._saver = tf.train.import_meta_graph(self._save_filename + "-last.meta")
    
    def _set_training_params(self, learning_rate=0.0001, hard_examples=False):
        """
        Build the training part of the network.

        :param learning_rate: The learning rate of the network.
        :param hard_examples: A boolean, true if the algorithm must show again the \
            pictures where the recognition failed
        """
        #check the graph has no training part
        self._graph.as_default()
        # get useful tensors
        x = tf.get_collection("x")[0]
        y_ = tf.get_collection("y_")[0]
        y_conv = tf.get_collection("y_conv")[0]
        # loss function
        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))
        # optimizer
        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_step = optimizer.minimize(cross_entropy)
        # evaluation
        with tf.variable_scope("Evaluation", reuse=tf.AUTO_REUSE):
            is_correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        if(hard_examples):
            with tf.variable_scope("Hardest_examples", reuse=tf.AUTO_REUSE):
                hard_index = tf.reshape(tf.where(tf.logical_not(is_correct)), [-1])
                hard_x = tf.gather(x, hard_index)
                hard_y_ = tf.gather(y_, hard_index)
        # save useful tensors
        graph = tf.get_default_graph()
        graph.add_to_collection("train_step", train_step)
        graph.add_to_collection("accuracy", accuracy)
        if(hard_examples):
            graph.add_to_collection("hard_x", hard_x)
            graph.add_to_collection("hard_y_", hard_y_)

        # initialize the network to save it
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.save(sess, self._save_filename+"-last")
        self._saver = saver
    
    def train(self, data, nb_iterations=10000, report_step=100, batch_size=100):
        """
        Train the network with the given data and save it.

        :param data: A set of data to train the network.
        :param nb_iterations: The number of batch to use for the training session.
        :param report_step: Run an evaluation of the accuracy each `report_step` batch.
        :param batch_size: The number of pictures in a batch.
        """
        # check params
        if(data is None):
            raise RuntimeError("Error: You did not specify data to train the network with.")
        # get useful tensors
        x = tf.get_collection("x")[0]
        y_ = tf.get_collection("y_")[0]
        keep_prob = tf.get_collection("keep_prob")[0]
        train_step = tf.get_collection("train_step")[0]
        accuracy = tf.get_collection("accuracy")[0]
        hard_examples = (len(tf.get_collection("hard_x")) > 0) # hard_examples
        if(hard_examples):
            hard_x = tf.get_collection("hard_x")[0]
            hard_y_ = tf.get_collection("hard_y_")[0]
        # Train the network in a session
        with tf.Session() as sess:
            self._saver.restore(sess, self._save_filename+"-last")
            print("Training started")
            start_time = time.time()
            for i in range(nb_iterations):
                batch_x, batch_y_ = data.train.next_batch(batch_size, shuffle=False)
                train_step.run(feed_dict={x: batch_x, y_: batch_y_, keep_prob: 0.5}) # if 0.5 is to low, modify it
                if(hard_examples): # train again with hard examples
                    hard_batch_x = hard_x.eval(feed_dict={x: batch_x, y_: batch_y_, keep_prob: 1})
                    hard_batch_y_ = hard_y_.eval(feed_dict={x: batch_x, y_: batch_y_, keep_prob: 1})
                    train_step.run(feed_dict={x: hard_batch_x, y_: hard_batch_y_, keep_prob: 0.5})
                if(i%report_step == 0):
                    train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y_, keep_prob: 1})
                    print("step %d : accuracy = %g" % (i, train_accuracy))
                    self._saver.save(sess, self._save_filename, global_step=i)
            # evaluation
            print("Training completed, evaluation...")
            print("Last accuracy =", accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1}))
            print("Model saved to:", self._saver.save(sess, self._save_filename+"-last"))
            print("Executed in: %.2f seconds"%(time.time() - start_time))

    def guess(self, pictures, dirname=DATABASE_DIR, filename=DATABASE_NAME):
        """
        Use the neuronal net to recognize the characters on the input pictures.

        :param pictures: A set (list) of 784 dimensions vectors (28x28 grayscale pictures).
        :param dirname: a str, the directory containing the mapping file.
        :param filename: a str, the prefixe of the mapping file (e.g: "ocr" in "ocr-mapping.txt")
        :return: A tuple : (list of characters, list of "probabilities")
        
        :raise: ValueError if pictures is not a list
        """
        # check parameters
        if(not isinstance(pictures, list)):
            raise ValueError("Error: pictures must be a list of pictures")
        self._graph.as_default()
        # get the interresting tensors
        x = tf.get_collection("x")[0]
        keep_prob = tf.get_collection("keep_prob")[0]
        y_output = tf.get_collection("y_output")[0]
        probability_output = tf.get_collection("probability_output")[0]
        # run a session
        index_list = None
        probability_list = None
        with tf.Session() as sess:
            self._saver.restore(sess, self._save_filename+"-last")
            index_list = y_output.eval(feed_dict={x: pictures, keep_prob: 1})
            probability_list = probability_output.eval(feed_dict={x: pictures, keep_prob: 1})
        return (NeuronalNetwork.get_char_from_index_list(index_list,dirname,filename), probability_list)

if __name__ == "__main__":
    data = NeuronalNetwork.load_data()
    # Train the network
    # nn = NeuronalNetwork() # set up the network with default parameters
    # nn.train(data)
    # Test the result of the training
    # nn = NeuronalNetwork(NeuronalNetwork.SAVE_FILE)
    # characters, _ = nn.guess([ data.test.images[i] for i in range(10, 63) ])
    # print(characters) # '0' to '9'
