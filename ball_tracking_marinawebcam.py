# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from matplotlib.colors import hsv_to_rgb
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
ap.add_argument("-t", "--track", action='store_true',
    help="enable tracking")
ap.add_argument("-s", "--save", action='store_true',
    help="enable tracking")
ap.add_argument("-o", "--output", type=str, default="output.avi",
	help="path to the output video file")
args = vars(ap.parse_args())

# list of tracked points
if args["track"]:
	pts = deque(maxlen=args["buffer"])  #How many old points do you keep to follow the track of the ball (red line in video)

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):	
	vs = VideoStream(src=0).start()
	frame = vs.read()
	frame_width = frame.shape[1]
	frame_height = frame.shape[0]

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
	frame_width = int(vs.get(3)) 
	frame_height = int(vs.get(4)) 

# allow the camera or video file to warm up
time.sleep(2.0)

# Get the frame width and height
   
size = (frame_width, frame_height) 

#This is for the video writer
if args["save"]:
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter(args["output"], fourcc, 10, size)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	#frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
	equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	hsv_equalized = cv2.cvtColor(equalized, cv2.COLOR_BGR2HSV)
	
    # filters low saturation colors (black, white and gray)
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	threshold_value = 25
	_, mask_inv = cv2.threshold(hsv_equalized[:, :, 1], threshold_value, 255, cv2.THRESH_BINARY)
	mask = cv2.bitwise_not(mask_inv)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None


	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour inb the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		circles = []
		for contour in cnts:
			cv2.drawContours(frame, [contour], -1, (160, 34, 119), 2)
			area = cv2.contourArea(contour)
			perimeter = cv2.arcLength(contour, True)

			threshold_black_value = 150
			_, mask_inv = cv2.threshold(hsv[:, :, 2], threshold_black_value, 255, cv2.THRESH_BINARY_INV)

			#cv2.imshow("mask", mask_inv)


			if int(perimeter) != 0 and ( np.any(hsv[:, :, 2][contour[:, :, 1], contour[:, :, 0]] < threshold_black_value) ) and 130000> area > 1050: # and int(area) != 201863: #Perimeter to avoid divide by 0, and area so it doesn't take the whole image as contour (this will depend on the image i think)
				circularity = 4 * np.pi * area / (perimeter * perimeter)
				circles.append((contour, circularity))

		# Choose the contour with the highest circularity
		circles.sort(key=lambda x: abs(x[1] - 1))  # Sort by circularity closest to 1
		
    		

		if len(circles) > 0 and abs(circles[0][1]-1) <  100: #This is a filter of enough circularity
			best_circle = circles[0][0]  # Select the contour with closest circularity to 1
			((x, y), radius) = cv2.minEnclosingCircle(best_circle)
			M = cv2.moments(best_circle)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			#if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points			
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	
	# loop over the set of tracked points
	if args["track"]:
		pts.appendleft(center)

		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue

			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	
	if args["track"]:
		out.write(frame)  # write the frame to the output video
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break


# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
if args["save"]:
	out.release()  # release the video writer
cv2.destroyAllWindows()
