# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from matplotlib.colors import hsv_to_rgb
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, opening, closing
import time

#Function
def compute_center(frame, h_threshold, s_threshold, v_threshold, k_erode, k_dilate, iterat_erode, iterat_dilate, ratio_limit, white_threshold, white_pixel_ratio_threshold):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each channel
    h_mask = hsv[:, :, 0] < h_threshold
    s_mask = hsv[:, :, 1] < s_threshold
    v_mask = hsv[:, :, 2] < v_threshold

    combined_mask = v_mask & s_mask & h_mask

    mask_inv = np.ones_like(combined_mask, dtype=np.uint8) * 255
    mask_inv[combined_mask] = 0
    mask = cv2.bitwise_not(mask_inv)

    #Erosion and dilation
    kernel_erode = np.ones((k_erode, k_erode), np.uint8)
    kernel_dilate = np.ones((k_dilate, k_dilate), np.uint8)

    mask1 = cv2.erode(mask, kernel_erode, iterat_erode) 
    mask2 = cv2.dilate(mask1, kernel_dilate, iterat_dilate) 

    # Find contours
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Make groups
    distances = 999* np.ones((len(contours), len(contours)))

    if len(contours) == 0:
        return 999, 999, 999

    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            # Find the centroid of each contour
            M1 = cv2.moments(contours[i])
            centroid1 = (int(M1['m10'] / M1['m00']), int(M1['m01'] / M1['m00']))

            M2 = cv2.moments(contours[j])
            centroid2 = (int(M2['m10'] / M2['m00']), int(M2['m01'] / M2['m00']))

            # Calculate Euclidean distance between centroids
            distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
            distances[i][j] = distance
            distances[j][i] = distance

    groups = []

    min_distances_indices = np.min(distances, axis=1)
    sourted_indices = np.argsort(min_distances_indices)

    if len(sourted_indices)>1:
        for i in sourted_indices:
            if not (any(i in sublist for sublist in groups)): #checks if that contour hasn't appear yet and calculate its own group
                group = []
                sorted_indices = np.argsort(distances[i])

                max_distance = distances[i, sorted_indices[1]] #Used to compare. The biggest one of the two closest

                group.extend(sorted_indices[:2].tolist())

                for other_index in sorted_indices[2:]:
                    for current_index in group:
                        if (distances[current_index, other_index] < ratio_limit * max_distance) and (other_index not in group):
                            group.append(other_index)
                
                groups.append(group)

    #Compute center
    old_best_ratio = 0

    for group in groups:
        group_contours = [contours[i] for i in group]
        group_contour = np.concatenate(group_contours)

        # Calculate minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(group_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Create a mask for whites inside the area encompassed by the circle
        mask_circle = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(mask_circle, center, radius, (255, 255, 255), thickness=-1)
        mask_binary = cv2.cvtColor(mask_circle, cv2.COLOR_BGR2GRAY)
        white_pixel_ratio = np.sum(frame[:, :, 2][mask_binary == 255] > white_threshold) / np.sum(mask_binary == 255)

        # Draw the circle only if the white pixel ratio is above the threshold
        if white_pixel_ratio > white_pixel_ratio_threshold and white_pixel_ratio > old_best_ratio:
            old_best_ratio = white_pixel_ratio
            best_radius = radius
            best_center = center
    
    if old_best_ratio == 0:
        return 999, 999, 999
    else:
        best_x, best_y = best_center

    return best_x, best_y, best_radius


#Parameters
iterat_erode = 1 #param
k_erode = 2 #param
iterat_dilate = 1 #param
k_dilate = 5 #param
ratio_limit = 1.4 #param
h_threshold = 100  # param
s_threshold = 65    # param
v_threshold = 110   # param
white_threshold = 200  #param
white_pixel_ratio_threshold = 0.2 #param


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

	# Calculation
	x_calc, y_calc, r_calc = compute_center(frame, h_threshold, s_threshold, v_threshold, k_erode, k_dilate, iterat_erode, iterat_dilate, ratio_limit, white_threshold, white_pixel_ratio_threshold)

	if x_calc != 999:					
		cv2.circle(frame, (x_calc, y_calc), r_calc, (0, 255, 255), 2)
		cv2.circle(frame, (x_calc, y_calc), r_calc, (0, 0, 255), -1)

	# update the points queue
	
	# loop over the set of tracked points
	if args["track"]:
		center = (x_calc, y_calc)
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
