# -*-coding:utf-8-*
"""
This file contains function for picture support
and compatbility with the neuronal network.
"""

# libraries
from PIL import Image
import numpy as np

def generate_noise_vector(size=(28, 28)):
    """
    Generate a random vector.

    :param size: tuple (width, height)
    :return: a vector generated randomly with values in [0,1[
    """
    return np.random.random_sample((size[0]*size[1],))

def picture2vector(pict):
    """
    This function returns the vector (neuronal network
    format) corresponding to the picture pict.

    :param pict: a Pillow picture
    :return: a 1-D vector in neuronal network input format
    """
    array = np.asarray(pict).flatten()
    return 1 - array/255

def vector2picture(vect, size=(28, 28)):
    """
    This function returns the picture
    corresponding to the picture vector vect
    (neuronal network format).

    :param vect: a 1-D vector in neuronal network input format
    :param size: a tuple (width, height) = dimensions of the picture
    :return: a Pillow picture
    """
    vect = (1 - vect) * 255
    vect = vect.reshape(size)
    return Image.fromarray(vect)

def view_vector_as_picture(vector, size=(28, 28)):
    """
    This function display a vector (in neuronal network input format)
    as a grayscale picture with the default picture viewer.

    :param vector: same format as the input of the neuronal network
    :type vetor: numpy.array
    :param size: tuple (width, height) of the picture
    """
    im = vector2picture(vector, size)
    im.show()

def isPictureFile(filename):
    """
    Return True if filename is a readable picture file.
    """
    try:
        Image.open(filename)
        return True
    except:
        pass
    return False