import cv2
import os
import numpy as numpy
from PIL import Image
import pickle
import sqlite3


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainingData.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = 'dataset'

def getProfile(id):
	conn = sqlite3.connect('FaceBase.db')
	cmd = cmd = "SELECT * FROM People WHERE ID="+str(id)
	cursor = conn.execute(cmd)
	profile = None

	for row in cursor:
		profile = row
	conn.close()
	return profile

cam = cv2.VideoCapture(0)

while True:
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,1.5,5)

	for x,y,w,h in faces:
		# id,conf = recognizer.predict(gray[y:y+h,x:x+w])
		# id = 5
		cv2.rectangle(gray,(x-50,y-50),(x+w+100,y+h+100),(0,255,0),3)
		# profile = getProfile(id)
		# if profile!=None:
		# 	print(profile)
	cv2.imshow('image',img)
	key = cv2.waitKey(1)
	if key==13:
		break
cam.release()
cv2.destroyAllWindows()