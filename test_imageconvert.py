# -*-coding:utf-8-*
"""
This file contains all the tests
relative to the imageconvert module.
"""

from imageconvert import generate_noise_vector, \
    vector2picture, picture2vector, view_vector_as_picture, \
    isPictureFile

def test_generate_noise_vector():
    size = (10,10)
    vect = generate_noise_vector(size)
    assert vect is not None
    assert len(vect) == size[0]*size[1]
    assert max(vect) < 1
    assert min(vect) >= 0

def test_picture_vector():
    vect = generate_noise_vector()
    pict = vector2picture(vect)
    vect2 = picture2vector(pict)
    assert len(vect) == len(vect2)
    for i in range(len(vect)):
        assert abs(vect[i] - vect2[i]) < 0.0001 # ==

#def test_view_vector_as_picture():
#    vect = generate_noise_vector()
#    view_vector_as_picture(vect)

def test_isPictureFile():
    assert not isPictureFile("imageconvert.py")
    assert isPictureFile("icon.ico")
    assert isPictureFile("test_pict.jpg")