from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# load the pre-trained model
face_detection_model = cv2.CascadeClassifier('./Model/Face_detection/haarcascade_frontalface_default.xml')

# load the face mask detector model from disk
mask_detection_model = load_model("./Model/Mask_detection/model.h5")

		
def detect_and_predict_mask(frame, face_detection_model, mask_detection_model):
	(h, w) = frame.shape[:2]
	# perform face detection
	image = np.array(frame, dtype='uint8')
	bboxes = face_detection_model.detectMultiScale(image)
	faces=[]
	locs=[]
	preds=[]
	# print bounding box for each detected face
	for box in bboxes:
		x1, y1, width, height = box.astype("int")
		x2, y2 = x1 + width, y1 + height
		face = frame[y1:y2, x1:x2]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)

		# add the face and bounding boxes to their respective
		# lists
		faces.append(face)
		locs.append((x1, y1, x2, y2))

		# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = mask_detection_model.predict(faces, batch_size=32)

	return (locs, preds)

video = VideoStream(src=0).start()
time.sleep(2.0)
    
while True:
	#extracting frames
	frame = video.read()
	frame = imutils.resize(frame, width=400)
	(locs, preds) = detect_and_predict_mask(frame, face_detection_model, mask_detection_model)
		
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(withoutMask,mask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
video.stop()