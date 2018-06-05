# -*-coding:utf-8-*
"""
This file contains the tests
relative to the extractor module.
"""

import sys
import os
from shutil import rmtree
from PIL import Image

from extractor import *

def save_chars(char_tab, name, image):
    # this function was removed from extractor.py because it's now useless there
    """
    Saves the points of interest in a folder

    Args:
        char_tab: list of Slicer object
        name: name of the original image file (relative path)
        image: the original image
    Raises:
        OSError: if the folder is already existing
    """
    newdir = os.path.abspath(os.path.join(os.getcwd(), name+"Sliced"))
    try:
        #creation of the folder
        os.mkdir(newdir)
        for i in range(0, len(char_tab)):
            r = char_tab[i].slicing(image)
            #save the character in the folder
            r.save(os.path.abspath(os.path.join(newdir, "{}{}".format(i, name))))
    except OSError:
        print("This folder already exists : {}".format(newdir), file=sys.stderr)

class TestExtractor:
    """
    This class tests the Slicer object.
    """
    def setup_class(self):
        self.image_filename = "test_extractor_img.png"
        self.image = Image.open(self.image_filename).convert("L")
    
    def teardown_class(self):
        if(os.path.exists(os.path.abspath(os.path.join(os.getcwd(), self.image_filename+"Sliced")))):
            rmtree(os.path.abspath(os.path.join(os.getcwd(), self.image_filename+"Sliced")))

    def test_stretching(self):
        img = stretching(self.image)
        (mini, maxi) = img.getextrema()
        assert mini == 0
        assert maxi >= 254

    def test_threshold(self):
        img = stretching(self.image)
        threshold(img, 128)
        p_a = img.load()
        w,h = img.size
        for i in range(w):
            for j in range(h):
                assert p_a[i, j] == 0 or p_a[i, j] == 255

    def test_init(self):
        slicer = Slicer(70, 42)
        assert slicer is not None

    def test_expend(self):
        img = stretching(self.image)
        threshold(img)
        pixel_access = img.load()
        w,h = img.size
        slicer = Slicer(70, 42)
        slicer.expend(70, 42, w, h, pixel_access)
        assert slicer.iMin == 63
        assert slicer.iMax == 76
        assert slicer.jMin == 41
        assert slicer.jMax == 57

    def test_sort(self):
        img = stretching(self.image)
        threshold(img)
        pixel_access = img.load()
        w,h = img.size
        s1 = Slicer(70, 42)
        s1.expend(70, 42, w, h, pixel_access)
        s2 = Slicer(86, 42)
        s2.expend(86, 42, w, h, pixel_access)
        tab = [ s2, s1 ]
        tab.sort()
        assert tab[0] is s1
        assert tab[1] is s2

    def test_slicing(self):
        img = stretching(self.image)
        threshold(img)
        pixel_access = img.load()
        w,h = img.size
        slicer = Slicer(70, 42)
        slicer.expend(70, 42, w, h, pixel_access)
        im = slicer.slicing(self.image, 28)
        assert im.size == (28, 28)
    
    def test_space_invader(self):
        img = stretching(self.image)
        threshold(img)
        pixel_access = img.load()
        w,h = img.size
        s1 = Slicer(70, 42)
        s1.expend(70, 42, w, h, pixel_access)
        s2 = Slicer(86, 42)
        s2.expend(86, 42, w, h, pixel_access)
        assert s1.space_invader(s2) == " "
        s2.iMin -= 10
        s2.iMax -= 10
        assert s1.space_invader(s2) == ""
        s2.jMin += 20
        s2.jMax += 20
        assert s1.space_invader(s2) == "\n"
    
    def test_str(self):
        img = stretching(self.image)
        threshold(img)
        pixel_access = img.load()
        w,h = img.size
        slicer = Slicer(70, 42)
        slicer.expend(70, 42, w, h, pixel_access)
        assert str(slicer) == "63 76 41 57"

    def test_detecting_chars(self):
        img = stretching(self.image)
        threshold(img)
        tab = detecting_chars(img)
        assert len(tab) == 3
    
    def test_char_detector(self):
        res = char_detector(self.image_filename) # it needs the nn
        assert res == "A B C\n"
