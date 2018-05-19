#!/usr/bin/python3.6
#-*- coding: utf-8

from PIL import Image
from Node import Slicer
import sys
import os
import math

def threshold(image,seuil):
	"""
	binarisation de l'image : chaque pixel > 128 devient 255
										   < 128 devient 0
	"""
	pixel_access = image.load()
	(width,height) = image.size
	for i in range (0,width-1) :
			for j in range (0,height-1) :
				if pixel_access[i,j] < seuil:
					pixel_access[i,j]=0
				else :
					pixel_access[i,j]=255
					

def detectingChars(image,width,height,charTab):
	"""
	fonction permettant la détection des caractères dans l'image
	en les stockant dans un tableau
	"""
	pixel_access = image.load()
	for i in range(0,width):
		for j in range (0,height):
			if(pixel_access[i,j]==0):
				s = Slicer(image,i,j)
				s.expend(i,j,width,height,pixel_access)
				
				charTab.append(s)

	
def stretching(img):
	"""
	étirement de l'histogramme de l'image pour améliorer
	le contraste
	"""
	(width,height) = img.size
	newimg = Image.new("L",(width,height),0)
	newpixel_access = newimg.load()
	pixel_access = img.load()
	(mini, maxi) = img.getextrema()
	
	coef = float(255)/float((maxi-mini))
	
	for i in range (0,width):
		for j in range (0,height):
			newpixel_access[i,j]=int(coef*(pixel_access[i,j]-mini))
		
	return newimg

def saveChars(charTab,nomPhoto,image):
	"""
	sauvegarde les point d'intérets relevés dans un dossier
	"""
	try:
		newdir = os.path.abspath(os.path.join(os.getcwd(),nomPhoto+"Sliced"))
		os.mkdir(newdir)
		
		for i in range(0,len(charTab)):
			r = charTab[i].slicing(image)
			r.save(os.path.abspath(os.path.join(newdir,"{}{}".format(i,nomPhoto))))
		
		print(newdir)
	except OSError:
		print("Dossier déjà existant")
		pass

		
def main():
	"""
	Programme principal
	"""
	try:	
		image = Image.open(sys.argv[1]).convert("L")
		img = stretching(image)
		(width,height) = img.size
		threshold(img,128);
		imageCopy = img.copy()
		charTab = []
		detectingChars(img,width,height,charTab)
		charTab.sort()
		saveChars(charTab,sys.argv[1],imageCopy)
		image.close()
		img.close()
		imageCopy.close()
		
	except IOError :
		print ("Fichier incorrect")



main();
