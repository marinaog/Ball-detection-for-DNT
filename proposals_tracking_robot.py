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
import itertools

#Functions
#Functions
#Visualize function
def visualize(image, gray = False, hsv_rep = False):
    plt.figure(figsize=(10, 5))

    if hsv_rep and not gray:
        img = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        plt.imshow(img) 

    elif not hsv_rep and gray:
        plt.imshow(image, cmap='gray')

    else:
        plt.imshow(image) 

    plt.show()


#Calculate distances between centers
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

#For creating groups of contours
def add_to_group(first, second, groups, sorted_indices, sorted_distances, pair_index):
    if len(groups)==0 or all(first != sublist[0] for sublist in groups): #checks if that contour hasn't appear yet and calculate its own group
        group = [first, second]

        for other_pair in sorted_indices:
            (other_first, other_second) = other_pair
            if (other_first == first and all(other_second != group_members for group_members in group)):
                other_pair_index = sorted_indices.index(other_pair)
                other_distance = sorted_distances[other_pair_index][2]
                if other_distance < ratio_limit*sorted_distances[pair_index][2]:
                    group.append(other_second)

            elif (other_second == first and all(other_first != group_members for group_members in group)):
                other_pair_index = sorted_indices.index(other_pair)
                other_distance = sorted_distances[other_pair_index][2]
                if other_distance < ratio_limit*sorted_distances[pair_index][2]:
                    other_distance < ratio_limit*sorted_distances[pair_index][2]
                    group.append(other_first)
        groups.append(group)


#For eliminating repetitive groups. For example: [1,2] and [2,1]
def are_sublists_equal(sublist1, sublist2):
    return sorted(sublist1) == sorted(sublist2)

def get_proposals(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each channel
    h_mask = hsv[:, :, 0] < h_threshold
    s_mask = hsv[:, :, 1] < s_threshold
    v_mask = hsv[:, :, 2] < v_threshold

    # Combine masks using logical AND operations
    combined_mask = v_mask & s_mask & h_mask

    # Create the final mask
    mask_inv = np.ones_like(combined_mask, dtype=np.uint8) * 255
    mask_inv[combined_mask] = 0
    mask = cv2.bitwise_not(mask_inv)

    #Erosion and dilation
    kernel_erode = np.ones((k_erode, k_erode), np.uint8)
    kernel_dilate = np.ones((k_dilate, k_dilate), np.uint8)
    mask1 = cv2.erode(mask, kernel_erode, iterat_erode) 
    mask2 = cv2.dilate(mask1, kernel_dilate, iterat_dilate) 

    # Find contours
    contours0, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #FOR VISUALIZATION: For drawing contours
    contours= []
    for contour in contours0:
        cv2.drawContours(frame, [contour], -1, (160, 34, 119), 2)
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter > min_perimeter: #newparam
            contours.append(contour)

    distances = 999* np.ones((len(contours), len(contours)))

    #Center of each contour area
    points = []
    dictionary = {}
    for i in range(len(contours)):
        M1 = cv2.moments(contours[i])
        centroid1 = (int(M1['m10'] / M1['m00']), int(M1['m01'] / M1['m00']))
        points.append(centroid1)
        dictionary[str(centroid1)]= i

    #Distances between contours
    distances = []
    for pair in itertools.combinations(points, 2):
        distance = calculate_distance(pair[0], pair[1])
        distances.append((pair[0], pair[1], distance))

    #Sorted by distance
    sorted_distances = sorted(distances, key=lambda x: x[2])
    sorted_indices = []
    for element in sorted_distances: #((108, 178), (76, 134), 54.405882034941776)
        first = dictionary[str(element[0])]
        second = dictionary[str(element[1])]
        sorted_indices.append((first, second))

    #Group of contours
    groups = []
    for pair in sorted_indices:
        pair_index = sorted_indices.index(pair)
        (first, second) = pair
        add_to_group(first, second, groups, sorted_indices, sorted_distances, pair_index)
        add_to_group(second, first, groups, sorted_indices, sorted_distances, pair_index)

    #Eliminate groups that are equal. For example [1,2] and [2,1]
    unique_groups = []
    for sublist in groups:
        if not any(are_sublists_equal(sublist, existing_sublist) for existing_sublist in unique_groups):
            unique_groups.append(sublist)

    #Choosing as first proposals those that pass requirements of having enough white inside
    proposals_int = {}
    new_unique_groups = []

    for group in unique_groups:
        group_contours = [contours[i] for i in group]
        # Concatenate all contours in the group
        group_contour = np.concatenate(group_contours)

        # Calculate minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(group_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Create a mask for the area encompassed by the circle
        mask_circle = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(mask_circle, center, radius, (255, 255, 255), thickness=-1)

        # Convert the mask to binary (0 or 255)
        mask_binary = cv2.cvtColor(mask_circle, cv2.COLOR_BGR2GRAY)

        # Calculate the ratio of white pixels in the circle area
        white_pixel_ratio = np.sum(frame[:, :, 2][mask_binary == 255] > white_threshold) / np.sum(mask_binary == 255)

        # Draw the circle only if the white pixel ratio is above the threshold
        if white_pixel_ratio > white_pixel_ratio_threshold: 
            proposals_int[str(group)]=([center, radius])
            new_unique_groups.append(group)

    #Avoid overlaping of proposals and select the one with most contours on it.
    proposals = []

    already_seen = {} #which element index: where with the biggest len seen for now
    len_group = {} #index of group: len

    for index, group in enumerate(new_unique_groups):
        for element in group:
            if str(element) in already_seen:
                seen_in_group = already_seen[str(element)]

                if str(seen_in_group) in len_group and len(group) > len_group[str(seen_in_group)]:
                    already_seen[str(element)] = index #here are the largest ones
                else:
                    len_group[str(element)] = len(group)
            else:
                already_seen[str(element)] = index
                len_group[str(element)] = len(group)

    final_groups = [new_unique_groups[index] for index in set(already_seen.values())]
    for group in final_groups:
        proposals.append(proposals_int[str(group)])

    final_proposal = []
    for proposal in proposals:
            [center, radius] = proposal
            if radius > min_radius:
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (255, 0, 0), -1)
                x, y = center
                final_proposal.append(proposal)
    return proposals



#Parameters
h_threshold = 120  # param
s_threshold = 80    # param
v_threshold = 80   # param
iterat_erode = 1 #param
k_erode = 2 #param
iterat_dilate = 1 #param
k_dilate = 5 #param
ratio_limit = 1.5 #param 
white_threshold = 75  #param
white_pixel_ratio_threshold = 0 #param

min_radius = 30 #newparam
min_perimeter = 25 #newparam

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
    proposals = get_proposals(frame)
    for proposal in proposals:
        [(x_calc, y_calc), r_calc] = proposal
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
