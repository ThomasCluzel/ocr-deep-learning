#-*- coding: utf-8

import sys
import os
import math
from PIL import Image

from imageconvert import picture2vector
from neuronallib import NeuronalNetwork


class Slicer:
    """
    The object which crop the characters of the given image
    """

    def __init__(self, i, j):
        """
        Args:
            i,j: position of the top left hand corner of the character 
        """
        self.iMin = i
        self.iMax = i
        self.jMin = j
        self.jMax = j

    def expend(self, i, j, width, height, p_a):
        """
        Define the bounding box of the characters with
        region growth
        
        Args :
            i,j : position of the current pixel
            width: height : size of the image
            p_a : current pixel access  
        """
        if(i >= 0 and j >= 0 and i < width and j < height and p_a[i, j] == 0):
            p_a[i, j] = 255
            if(i < self.iMin):
                self.iMin = i
            elif(i > self.iMax):
                self.iMax = i
            if(j < self.jMin):
                self.jMin = j
            elif(j > self.jMax):
                self.jMax = j
            #recursive call on adjacent pixels
            self.expend(i-1, j, width, height, p_a)
            self.expend(i, j+1, width, height, p_a)
            self.expend(i+1, j, width, height, p_a)
            self.expend(i, j-1, width, height, p_a)

    def __lt__(self, reg2):
        """
        Redefinition of the operation < between two Slicer
        in order to sort a list of Slicer
        
        Args:
            reg2: a Slicer object
            
        Returns:
            a boolean 
        """
        if(abs(self.jMin-reg2.jMin) < (self.jMax-self.jMin)/2):
            return self.iMin < reg2.iMin
        else:
            return self.jMin < reg2.jMin

    def slicing(self, image, size=28):
        """
        Slicing of the character in an image of size px * size px
        
        Args:
            image: the orginal PIL image
            size: the size of the square region around the character
            
        Returns:
            a size*size PIL image
        """
        w = self.iMax-self.iMin+1
        h = self.jMax-self.jMin+1
        maxi = max(w, h)
        #creation of the white result image
        result = Image.new("L", (maxi, maxi), 255)
        #crop the bounding box in the original image
        image = image.crop((self.iMin, self.jMin, self.iMax+1, self.jMax+1))
        #paste the character in the result image
        result.paste(image, ((maxi-w)//2, (maxi-h)//2))
        #resize the result image with the same proportion as in the original
        return result.resize((size, size), Image.BICUBIC)

    def space_invader(self, slicer2):
        """
        Detects spaces or new lines between 2 characters
        
        Args:
            slicer2: the next slicer in the list
            
        Returns:
            a string which contains a space, a \n or nothing
        """
        ret = ""
        if(abs(self.jMin-slicer2.jMin) > 15):
            ret = "\n"
        elif((slicer2.iMin-self.iMax) > (self.iMax-self.iMin)):
            ret = " "
        return ret

    def __str__(self):
        """
        Redefinition of the Slicer's string representation 
        """
        return "{} {} {} {}".format(self.iMin, self.iMax, self.jMin, self.jMax)


def threshold(image, value=128):
    """
    Thresholding of the image

    Args:
        image: a PIL image
        value: an integer between 0 and 255
    """
    pixel_access = image.load()
    (width, height) = image.size
    for i in range(0, width-1):
            for j in range(0, height-1):
                if pixel_access[i, j] < value:
                    pixel_access[i, j] = 0
                else:
                    pixel_access[i, j] = 255


def detecting_chars(image):
    """
    Function which detects the chars in the image
    and generates a list of Slicers objects

    Args:
        image: a PIL image
        
    Returns:
        list which stores the Slicers objects
    """
    (width, height) = image.size
    charTab = []
    pixel_access = image.load()
    for i in range(0, width):
        for j in range(0, height):
            #if the current pixel is black
            if(pixel_access[i, j] == 0):
                s = Slicer(i, j)
                #we create a new Slicer and we expend it
                s.expend(i, j, width, height, pixel_access)
                charTab.append(s)
    return charTab


def stretching(img):
    """
    Stretching the histogram to improve the contrast

    Args:
        img : a PIL image
        
    Returns :
        the stretched image
    """
    (width, height) = img.size
    newimg = Image.new("L", (width, height), 0)
    newpixel_access = newimg.load()
    pixel_access = img.load()
    (mini, maxi) = img.getextrema()
    #calculation of the stretching coeficient
    coef = float(255)/float((max(maxi-mini, 1)))
    for i in range(0, width):
        for j in range(0, height):
            #calculation of the new pixel's value
            newpixel_access[i, j] = int(coef*(pixel_access[i, j]-mini))
    return newimg


def save_chars(charTab, name, image): # TODO: remove this function from this file (move it to test_extractor.py)
    """
    Saves the points of interest in a folder

    Args:
        charTab: list of Slicer object
        name: name of the original image file (relative path)
        image: the original image
    Raises:
        OSError: if the folder is already existing
    """
    newdir = os.path.abspath(os.path.join(os.getcwd(), name+"Sliced"))
    try:
        #creation of the folder
        os.mkdir(newdir)
        for i in range(0, len(charTab)):
            r = charTab[i].slicing(image)
            #save the character in the folder
            r.save(os.path.abspath(os.path.join(newdir, "{}{}".format(i, name))))
        print(newdir)
    except OSError:
        print("This folder already exists : {}".format(newdir), file=sys.stderr)
	


def char_detector(img_filename):
    """
    Crop the image and recognize the characters
    with a neuronal network

    Args:
        img_filename: image filename
        
    Returns:
        a string that matches the text on the input picture
        
    Raises:
        IOError: if the given file does not exist
    """
    try:
        image = Image.open(img_filename).convert("L")
    except IOError:
        raise IOError("Incorrect filename: %s is not a picture" % img_filename)
    img = stretching(image)
    threshold(img)
    charTab = detecting_chars(img)
    charTab.sort()
    nn = NeuronalNetwork(NeuronalNetwork.SAVE_FILE)
    result = ""
    vectorTab = [picture2vector(i.slicing(image)) for i in charTab]
    chars,_ = nn.guess(vectorTab) # use the nn to turn pictures into characters
    for i in range(len(charTab)-1):
        result += chars[i] # append the character to text
        result += charTab[i].space_invader(charTab[i+1]) # append a space if necessary
    result += chars[len(chars)-1] + "\n" # don't forget the last one
    image.close()
    img.close()
    return result


if __name__ == "__main__":
    if(len(sys.argv) >= 2):
        print(char_detector(sys.argv[1]))
    else:
        print("Gimme a picture as first param, please", file=sys.stderr)
