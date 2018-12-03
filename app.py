from flask import Flask, request, flash, json
import os
from flask_script import Manager, Server
from werkzeug.utils import secure_filename
import pathlib
import random
from gevent.pywsgi import WSGIServer
from imutils import paths
import random
import cv2
import json
import os
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import argparse
from keras import backend as K
import tensorflow as tf

import socket

app = Flask(__name__)

app.secret_key = os.urandom(94743)

UPLOAD_FOLDER = os.path.basename('Uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

TEST_FOLDER = os.path.basename('CustomTestImages')
app.config['TEST_FOLDER'] = TEST_FOLDER


RESIZE_DIM=50 
TARGET_SIZE = 400



def loadModel(jsonPath,weightsPath):

	jsonfile = open(jsonPath,'r')
	loaded_model_json = jsonfile.read()
	jsonfile.close()

	print(type(loaded_model_json))

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weightsPath)
	return loaded_model



def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)



from skimage import exposure
from skimage import data, io, filters

def connectedComp(img):

	img = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY)[1]  # ensure binary
	ret, labels = cv2.connectedComponents(img)

	# Map component labels to hue val
	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

	# set bg label to black
	labeled_img[label_hue==0] = 0

	return img

def morph(input_image):
	input_image = cv2.threshold(input_image, 254, 255, cv2.THRESH_BINARY)[1]
	input_image_comp = cv2.bitwise_not(input_image)  # could just use 255-img

	kernel1 = np.array([[0, 0, 0],
						[0, 1, 0],
						[0, 0, 0]], np.uint8)
	kernel2 = np.array([[1, 1, 1],
						[1, 0, 1],
						[1, 1, 1]], np.uint8)

	hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_ERODE, kernel1)
	hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
	hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
	return hitormiss



def remove_isolated_pixels(image):
	connectivity = 8

	output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

	num_stats = output[0]
	labels = output[1]
	stats = output[2]

	new_image = image.copy()

	for label in range(num_stats):
		if stats[label,cv2.CC_STAT_AREA] == 1:
			new_image[labels == label] = 0

	return new_image

def process_image(imagePath):
	image = cv2.imread(imagePath)

	TARGET_SIZE = 400

	centerX = image.shape[0]//2
	centerY = image.shape[1]//2



	startX = max(0,centerX-TARGET_SIZE//2)
	startY = max(0,centerY-TARGET_SIZE//2)


	# img = adjust_gamma(image,.5)

	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img = cv2.bilateralFilter(img,33,33,99)
	# equ = cv2.equalizeHist(img)
	equ = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 2)
	equ = connectedComp(equ)
	# thresh_gray = connectedComp(thresh_gray)
	# equ = cv2.addWeighted(equ,1.5,img,-0.5,0)


	cropped = equ[startX:startX+TARGET_SIZE,startY:startY+TARGET_SIZE]

	
	resized = cv2.cvtColor(cropped,cv2.COLOR_GRAY2RGB)
	resized = cv2.resize(resized, (RESIZE_DIM,RESIZE_DIM))
	return cv2.fastNlMeansDenoising(resized,None,10,7,21)


loaded_model = loadModel('./Classifier/trained_model.json','./Classifier/trained_model_weights.h5')
loaded_model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
graph = tf.get_default_graph()


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(["jpg","png","svg","jpeg"])


def checkApplicationKey(request):
	appkey = request.headers.get("Appkey")
	if((appkey is None) or (appkey!="E78E2433C18FFA9E5CF85DF1DE1EC")):
		return False
	return True

def checkFileExistence(request):
	if 'file' not in request.files:
		flash("No Files Part");
		return False
	return True

def checkFileName(file):
	if(file.filename==''):
		return False
	return True


def classifyThisImage(imagePath):
	print(imagePath)

	processedImage = process_image(imagePath)
	cv2.imwrite(imagePath+"1"+".png",processedImage)
	processedImage = np.expand_dims(processedImage, axis=0)
	processedImage = np.array(processedImage,dtype='float')/255.0

	global graph
	with graph.as_default():
		predictions = loaded_model.predict(processedImage)
		print(processedImage.shape)
	print(predictions)
	print(predictions.argmax(axis=1))
	print(len(predictions))
	print(predictions)
	return str(predictions.argmax(axis=1)[0]),str(predictions[0][predictions.argmax(axis=1)[0]]*100.0) 


def handleFile(request,requestCode):
	if(checkApplicationKey(request)==False or checkFileExistence(request)==False):
		return json.dumps({'success': False}), 400, {'ContentType': 'application/json'}


	#valuidate filename not empty
	file = request.files['file']
	if(checkFileName(file)==False):
		return json.dumps({'success': False}), 400, {'ContentType': 'application/json'}


	body = request.form
	fileName = body['fileName']
	label = body['label']

	
	#ordinary file upload function
	save_folder = ""
	if(requestCode==1):
			#validate applicationkey and fileexistence
		save_folder = os.path.join(app.config['UPLOAD_FOLDER'],label)
		print(save_folder)
	elif(requestCode==2):
		save_folder = os.path.join(app.config['TEST_FOLDER'])

	pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

	CURRENT_FILE_PATH = ""
	if(file and allowed_file(fileName)):
		filename = secure_filename(fileName)
		f = os.path.join(save_folder, str(random.getrandbits(64))+".jpg")
		file.save(f)
		CURRENT_FILE_PATH = f
	else:
		return json.dumps({'success': False}), 400, {'ContentType': 'application/json'}


	if(requestCode==1):
		contribution_file = os.path.join("contributions.csv")
		username = body['username']
		useremail = body['useremail']
		userphone = body['userphone']
		userorganization = body['userorganization']
		print(str(fileName)+","+str(label)+","+str(username)+","+str(useremail)+","+str(userphone)+","+str(userorganization),
		file=open(contribution_file,"a"))
		return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}

	elif(requestCode==2):
		detectedLabel,conf = classifyThisImage(CURRENT_FILE_PATH)
		return json.dumps({'success': True, 'detectedLabel':detectedLabel+" \nconfidence: "+conf+"%"}), 200, {'ContentType': 'application/json'}



@app.route('/classify',methods=['POST'])
def classify_image():
	return handleFile(request,2)



@app.route('/upload', methods=['POST'])
def upload_file():
	return handleFile(request,1)




if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='You can provide IP and Port through cmd')
	parser.add_argument('-H','--host', help='IP address of the host', required=False)
	parser.add_argument('-P','--port', help='Corresponding port number', required=False)
	args = vars(parser.parse_args())

	host=''

	if(args['host']):
		try:
			socket.inet_aton(args['host'])
			host=args['host']
		except socket.error:
			print("IP address is invalid, starting on http://localhost")
			host=''

	port=8080
	if(args['port']):
		port = int(args['port'])

	app.run(host,port)

	# print("SERVING FOREVER on "+host+":"+str(port)+"/")
	# http_server = WSGIServer((host,port), app)
	# http_server.serve_forever()