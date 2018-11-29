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



def process_image(imagePath):
	image = cv2.imread(imagePath)
	TARGET_SIZE = 400

	centerX = image.shape[0]//2
	centerY = image.shape[1]//2

	startX = max(0,centerX-TARGET_SIZE//2)
	startY = max(0,centerY-TARGET_SIZE//2)

	cropped = image[startX:startX+TARGET_SIZE,startY:startY+TARGET_SIZE]

	gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) # convert to grayscale
	# threshold to get just the signature
	thresh_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 2)

	r = 200.0 / thresh_gray.shape[1]
	dim = (200, int(thresh_gray.shape[0] * r))
	 
	# perform the actual resizing of the image and show it
	resized = cv2.resize(thresh_gray, dim, interpolation = cv2.INTER_AREA)
	resized = cv2.cvtColor(resized,cv2.COLOR_GRAY2RGB)
	resized = cv2.resize(resized, (RESIZE_DIM,RESIZE_DIM))
	return resized




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
	processedImage = np.expand_dims(processedImage, axis=0)
	processedImage = np.array(processedImage,dtype='float')/255.0

	global graph
	with graph.as_default():
		predictions = loaded_model.predict(processedImage)
		print(processedImage.shape)
	print(predictions)
	print(predictions.argmax(axis=1))
	return str(predictions.argmax(axis=1)[0]) 


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
		detectedLabel = classifyThisImage(CURRENT_FILE_PATH)
		return json.dumps({'success': True, 'detectedLabel':detectedLabel}), 200, {'ContentType': 'application/json'}



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