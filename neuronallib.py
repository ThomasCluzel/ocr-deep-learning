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
characters = nn.guess(pictures)
"""

# TODO :
# - rework the dataset provider (concatenate columns to the mnist dataset is heavy)
#   - use the datamanger.py
#   - use emnist-byclass (use a prefix and make the end default for the 4 cases)
#   - use the mapping file (with the correct prefix) in get_char_from_index
# - find pictures that are not noise nor digits to train the 11th (last) class

# libraries
import os
import time
import numpy as np
import tensorflow as tf
from imageconvert import generate_noise_vector


class NeuronalNetwork:
    """
    This class manages the neuronal network.
    It can create (from scratch) a neuronal netwok, load an already
    existing network, use a network to guess a character, train a network
    and save a network.
    """

    # Class attributes and methodes
    SAVE_FILE = os.path.dirname(os.path.realpath(__file__)) + "/cnn/model"
    NB_OUTPUTS_CLASSES = 11
    # TODO : add the prefixes here and the default end ?

    @staticmethod
    def get_char_from_index_list(index_list):
        """
        Returns the list of character corresponding to the index
        in the index list compute by the neuronal network.

        :param index_list: List return by the evaluation of the
            output node of the graph.
        :return: A list of characters "read" by the network.
        """
        def get_char_from_index(index):
            if(index>=NeuronalNetwork.NB_OUTPUTS_CLASSES-1):
                return ' ' # in case the character isn't recognized
            return chr(index + ord('0'))
        return list(map(get_char_from_index, index_list))

    def __init__(self, save_filename=None):
        """
        Load the network saved in the given file if any
        or generate a network to train.

        :param save_filename: The name of the file containing the saved
            network to load. If None a new graph is created.
        :type save_filename: str
        """
        # load or generate ?
        if save_filename is None:
            # Create the network
            self._graph = tf.Graph()
            self._create_network()
        else:
            # load the saved network
            self.load(save_filename)
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
            y_ = tf.placeholder(tf.float32, [None, NeuronalNetwork.NB_OUTPUTS_CLASSES]) # 10 digits + 1 nothing
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
            W_fc2 = weight_variable([1024, NeuronalNetwork.NB_OUTPUTS_CLASSES])
            b_fc2 = bias_variable([NeuronalNetwork.NB_OUTPUTS_CLASSES])
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

    def load(self, save_filename=SAVE_FILE):
        """
        Load a network saved in a file.

        :param save_filename: The name of the file containing the saved
            network to load.
        :type save_filename: str
        """
        if(save_filename is None):
            raise ValueError("Error: the file name is empty")
        self._saver = tf.train.import_meta_graph(save_filename + "-last.meta")

    def train(self, data, save_filename=SAVE_FILE, learning_rate=0.0001,
            nb_iterations=10000, report_step=100, batch_size=100,
            hard_examples=False, compute_time=True):
        """
        Train the network with the given data and save it in save_filename.

        :param data: A set of data to train the network.
        :param save_filename: The beginning of a filename with directory. \
            example : current_dir + "cnn/model"
        :param learning_rate: The learning rate of the network.
        :param nb_iterations: The number of batch to use for the training session.
        :param report_step: Run an evaluation of the accuracy each `report_step` batch.
        :param batch_size: The number of pictures in a batch.
        :param hard_examples: A boolean, true if the algorithm must show the again the \
            pictures where the recognition failed
        :param compute_time: A boolean, true if the computation time must be displayed.
        """
        # check parameters
        assert data is not None
        assert save_filename is not None
        self._graph.as_default()
        # get useful tensors
        x = tf.get_collection("x")[0]
        y_ = tf.get_collection("y_")[0]
        keep_prob = tf.get_collection("keep_prob")[0]
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
        # In case where we want to train the model more (save useful nodes)
        graph = tf.get_default_graph()
        graph.add_to_collection("train_step", train_step)
        graph.add_to_collection("accuracy", accuracy)

        # the session to train the model
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            print("Training started")
            if(compute_time):
                start_time = time.time()
            for i in range(nb_iterations):
                batch_x, batch_y_ = data.train.next_batch(batch_size, shuffle=False)
                # modify the labels from 10 to 11
                batch_y_ = np.hstack( (batch_y_, [[0]*(NeuronalNetwork.NB_OUTPUTS_CLASSES-10) for i in range(batch_size)]) )
                # add some noisy pictures to the batch for the 11th class
                batch_x = np.vstack( (batch_x, [ generate_noise_vector() for i in range(batch_size//10) ]) )
                batch_y_ = np.vstack( (batch_y_, [ [0]*10+[1]*(NeuronalNetwork.NB_OUTPUTS_CLASSES-10) for i in range(batch_size//10) ]) )
                # then train (this is the next line that really train the network)
                train_step.run(feed_dict={x: batch_x, y_: batch_y_, keep_prob: 0.5}) # if 0.5 is to low, modify it
                if(hard_examples):# train again with hard examples
                    hard_batch_x = hard_x.eval(feed_dict={x: batch_x, y_: batch_y_, keep_prob: 1})
                    hard_batch_y_ = hard_y_.eval(feed_dict={x: batch_x, y_: batch_y_, keep_prob: 1})
                    train_step.run(feed_dict={x: hard_batch_x, y_: hard_batch_y_, keep_prob: 0.5})
                if(i%report_step == 0):
                    train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y_, keep_prob: 1})
                    print("step %d : accuracy = %g" % (i, train_accuracy))
                    saver.save(sess, save_filename, global_step=i)
            # evaluation
            images = data.test.images
            labels = data.test.labels # modify the labels to have 11 output labels
            labels = np.hstack( (labels, [[0]*(NeuronalNetwork.NB_OUTPUTS_CLASSES-10) for i in range(labels.shape[0])]) )
            # and add noisy pictures in the evaluation set
            images = np.vstack( (images, [ generate_noise_vector() for i in range(100) ]) )
            labels = np.vstack( (labels, [ [0]*10+[1]*(NeuronalNetwork.NB_OUTPUTS_CLASSES-10) for i in range(100) ]) )
            # then evaluate
            print("Training completed, evaluation...")
            print("Last accuracy =", accuracy.eval(feed_dict={x: images, y_: labels, keep_prob: 1}))
            print("Model saved to:", saver.save(sess, save_filename+"-last"))
            if(compute_time):
                print("Executed in: %.2f seconds"%(time.time() - start_time))

    def guess(self, pictures, save_filename=SAVE_FILE):
        """
        Return the list of characters corresponding to the characters
        written on the pictures.

        :param pictures: A set of 784 dimensions vectors (28x28 grayscale pictures).
        :param save_filename: The file (directory/file) of the graph.
        """
        # check parameters
        assert isinstance(pictures, list)
        assert save_filename is not None
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
            self._saver.restore(sess, save_filename+"-last")
            index_list = y_output.eval(feed_dict={x: pictures, keep_prob: 1})
            probability_list = probability_output.eval(feed_dict={x: pictures, keep_prob: 1})
        return (NeuronalNetwork.get_char_from_index_list(index_list), probability_list)

def create_and_train_model():
    """
    Create a neuronal network and train it with default data.
    """
    nn = NeuronalNetwork()
    # download MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # train the network
    #nn.train(mnist, report_step=1000)
    nn.train(mnist, learning_rate=0.001, nb_iterations=100, report_step=50)

def test_model(nb_test=20):
    # load the graph
    nn = NeuronalNetwork(NeuronalNetwork.SAVE_FILE)
    # download MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # test some pictures
    pict = [ mnist.test.images[i] for i in range(nb_test-1) ] + [ generate_noise_vector() ]
    correct = NeuronalNetwork.get_char_from_index_list([ mnist.test.labels[i].tolist().index(1) for i in range(nb_test-1) ]) + [' ']
    numbers, prob = nn.guess(pict)
    rate = sum(list(map(lambda x, y: x==y and 1 or 0, numbers, correct))) / nb_test
    print("Correct rate (%d tries) is: %f" % (nb_test, rate), "\n  Nums  ==>", numbers, "\n  Prob  ==>", prob.round(2))

if __name__ == "__main__":
    print("1 - create and train a new network model", "2 - test a saved model", "other - exit", sep="\n")
    check = int(input())
    if(check == 1):
        create_and_train_model()
    elif(check == 2):
        test_model()
