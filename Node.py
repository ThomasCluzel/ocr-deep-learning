#!/usr/bin/python3.6
#-*- coding: utf-8

from PIL import Image


class Slicer(object):
	"""
	objet découpant les caractères de l'image
	"""
	def __init__(self,image,i,j):
		self.image = image
		self.iMin = i
		self.iMax = i
		self.jMin = j
		self.jMax = j
	
	def expend(self,i,j,width,height,p_a):
		"""
		définition d'une "bounding box" autour du caractère par
		croissance de région
		"""
		if(i>=0 and j>=0 and i<width-1 and j<height-1 and p_a[i,j]==0):
			p_a[i,j]=255
			if(i<self.iMin):
				self.iMin = i
			elif(i>self.iMax):
				self.iMax = i
			if(j<self.jMin):
				self.jMin = j
			elif(j>self.jMax):
				self.jMax = j
			#appels récursifs sur les pixels adjacents du pixel courant
			self.expend(i-1,j,width,height,p_a)
			self.expend(i,j+1,width,height,p_a)
			self.expend(i+1,j,width,height,p_a)
			self.expend(i,j-1,width,height,p_a)

		
		
	def __lt__(self, reg2):
		"""
		redéfinition de l'opération de comparaison < entre Slicer
		pour pouvoir trier un tableau d'objets Slicer
		"""
		if(abs(self.jMin-reg2.jMin)<8):
			return self.iMin<reg2.iMin
		else:
			return self.jMin<reg2.jMin
	
	def slicing(self,image):
		"""
		découpage du caractère en une image de 28px*28px
		"""
		w = self.iMax-self.iMin
		h = self.jMax-self.jMin
		maxi = max(w,h,28)
		result = Image.new("L",(maxi,maxi),255)
		image = image.crop((self.iMin,self.jMin,self.iMax+1,self.jMax+1))
		result.paste(image,(int(maxi/2 - w/2 -1),int(maxi/2-h/2 -1),int(maxi/2-w/2+w),int(maxi/2-h/2+h)))
		result.thumbnail((28,28), Image.ANTIALIAS)
		return result
		
	def __str__(self):
		"""
		redéfinition de la représentation en string d'un objet Slicer 
		"""
		return "{} {} {} {}".format(self.iMin,
		self.iMax,
		self.jMin,
		self.jMax)
		
	
		
	
