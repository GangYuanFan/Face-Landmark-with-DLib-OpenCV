from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
fname = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(fname)
cap = cv2.VideoCapture(0)
def get_landmarks(gray):
	try:
		# detect faces in the grayscale image
		rects = detector(gray, 1)

		# loop over the face detections
		for (i, rect) in enumerate(rects):
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			xlist = []
			ylist = []
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
				xlist.append(x)
				ylist.append(y)

			xmean = np.mean(xlist)
			ymean = np.mean(ylist)
			print("xmean = %d, and ymean = %d" % (xmean, ymean))
			xcentral = [(x - xmean) for x in xlist]
			ycentral = [(y - ymean) for y in ylist]
			cv2.circle(image, (int(xmean), int(ymean)), 1, (255, 0, 0), -1)

			landmarks_vectorised = []
			# landmarks_vectorised (x, y, length(point2central), angle)
			for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
				landmarks_vectorised.append(w)
				landmarks_vectorised.append(z)
				meannp = np.asarray((ymean, xmean))
				coornp = np.asarray((z, w))
				dist = np.linalg.norm(coornp - meannp)  # find norm of vector
				landmarks_vectorised.append(dist)
				landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

			for _ in range(len(landmarks_vectorised)):
				if _ % 4 == 0:
					cv2.line(image, (int(landmarks_vectorised[_]), int(landmarks_vectorised[_+1])), (int(xmean), int(ymean)), (0, 255, 0), 1)
			return landmarks_vectorised, image
	except:
		print("no faces!!!")

while True:
	ret, image = cap.read()
	if ret:
		# image = imutils.resize(image, width=500)
		image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation=cv2.INTER_LINEAR)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		landmark = get_landmarks(gray)
		if landmark != None:
			landmark, image = np.array(landmark)
		cv2.imshow("Output", image)
		cv2.waitKey(1)
