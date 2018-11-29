from imutils import paths
import random
import cv2
import json
import os
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import pickle


def loadModel(jsonPath,weightsPath):
	jsonfile = open(jsonPath,'r')
	loaded_model_json = jsonfile.read()
	jsonfile.close()

	print(type(loaded_model_json))

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weightsPath)

	return loaded_model

loaded_model = loadModel('./trained_model.json','./trained_model_weights.h5')
loaded_model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


pickle.dump(loaded_model,open("model.pkl",'wb'))

