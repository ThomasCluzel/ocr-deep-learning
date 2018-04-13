# -*-coding:utf-8-*
"""
This file contains some tests
relative to the datamanager module.

Actually the only function I modified is read_data_sets
and I kept the others as is.
"""

from datamanager import read_data_sets

def test_read_data_sets():
    data = read_data_sets(train_dir = "OCR_data",
                              train_images = 'ocr-train-images-idx3-ubyte.gz',
                              train_labels = 'ocr-train-labels-idx1-ubyte.gz',
                              test_images = 'ocr-test-images-idx3-ubyte.gz',
                              test_labels = 'ocr-test-labels-idx1-ubyte.gz',
                              num_classes = 63,
                              one_hot=True)
    assert data is not None
    assert data.train is not None
    assert data.train.images is not None
    assert data.train.labels is not None
    assert len(data.train.images) == len(data.train.labels)
    assert data.test is not None
    assert data.test.images is not None
    assert data.test.labels is not None
    assert len(data.test.images) == len(data.test.labels)