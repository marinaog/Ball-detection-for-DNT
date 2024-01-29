#Look for best parameters
import struct
import imghdr


#Function needed to obtain width and height of image
def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height
    
# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
from skimage.morphology import erosion, dilation, opening, closing
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image
from IPython.display import Image as IPImage, display



def compute_center(image_path, h_threshold, s_threshold, v_threshold, k_erode, k_dilate, iterat_erode, iterat_dilate, ratio_limit, white_threshold, white_pixel_ratio_threshold):
    frame = cv2.imread(image_path)
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
                

#LOAD TEXT FILES
import os
with open('selected_image_names.txt', 'r') as file_imgnames:
    dataset = [line.strip() for line in file_imgnames]

images_info_dict = {}

lenn = len(dataset)
i=0
for image in dataset: 
    i+=1
    image_png = image
    
    image_nam = image_png.split('.')
    image_name = image_nam[0]

    label_file_path = 'selected_labels/' + str(image_name) + '.txt'
    if os.path.exists(label_file_path) and os.path.exists('selected_images/' + str(image_png)):
        with open(label_file_path, 'r') as file2:
            for line in file2:
                parts = line.strip().split()
                if parts[0] == '0': #means that is of class ball 0 0.4921875 0.264583333333 0.1875 0.25 [x_real, y_real, w_box, h_box]
                    width, height = get_image_size('selected_images/' + str(image_png))
                    print(int(float(parts[1])*width))
                    images_info_dict[image_png] = {'x_real': int(float(parts[1])*width), 'y_real' :  int(float(parts[2])*height), 'w_box' :  int(float(parts[3])*width), 'h_box' :  int(float(parts[4])*height)}
    
print(images_info_dict)

print('Number of images: ', len(images_info_dict))


#SET RANGE OF PARAMETERS WITH WHICH WE WILL PLAY
#lims [min, max, steps]
lim_h_threshold = [50, 120, 10] #[20, 100, 10] 
lim_s_threshold = [60, 75, 5] #[25, 75, 5] 
lim_v_threshold = [65, 75, 5] #[25, 75, 5]

lim_k_erode = [6, 7, 2] 
lim_k_dilate = [5, 6, 2] 

lim_iterat_erode = [1, 2, 1] 
lim_iterat_dilate = [1, 2, 1] 

lim_ratio_limit = [1.75, 2, 0.25]  #[1.2, 2, 0.25] 

lim_white_threshold = [125, 175, 25] 
lim_white_pixel_ratio_threshold = [0, 0.045, 0.015] #[0, 0.045, 0.005]


#NUMBER OF COMBINATIONS
num_combinations = (
    len(range(lim_h_threshold[0], lim_h_threshold[1], lim_h_threshold[2])) *
    len(range(lim_s_threshold[0], lim_s_threshold[1], lim_s_threshold[2])) *
    len(range(lim_v_threshold[0], lim_v_threshold[1], lim_v_threshold[2])) *
    len(range(lim_k_erode[0], lim_k_erode[1], lim_k_erode[2])) *
    len(range(lim_k_dilate[0], lim_k_dilate[1], lim_k_dilate[2])) *
    len(range(lim_iterat_erode[0], lim_iterat_erode[1], lim_iterat_erode[2])) *
    len(range(lim_iterat_dilate[0], lim_iterat_dilate[1], lim_iterat_dilate[2])) *
    len(np.arange(lim_ratio_limit[0], lim_ratio_limit[1], lim_ratio_limit[2])) *
    len(range(lim_white_threshold[0], lim_white_threshold[1], lim_white_threshold[2])) *
    len(np.arange(lim_white_pixel_ratio_threshold[0], lim_white_pixel_ratio_threshold[1], lim_white_pixel_ratio_threshold[2]))
)

print("Number of combinations:", num_combinations*len(images_info_dict))


combinations = [] #each element will be a list with the parameters of the combination #index of the element. total length = number of combinations
puntuations = [] #each element will be a list of length the number of images in the datalist with 1 when correct and 0 when incorrect. total length = number of combinations x number of images

i_time = time.time()
i = 0
#Accuracy calculation
for h_threshold in range(lim_h_threshold[0], lim_h_threshold[1], lim_h_threshold[2]):
    for s_threshold in range(lim_s_threshold[0], lim_s_threshold[1], lim_s_threshold[2]):
        for v_threshold in range(lim_v_threshold[0], lim_v_threshold[1], lim_v_threshold[2]):
            for k_erode in range(lim_k_erode[0], lim_k_erode[1], lim_k_erode[2]):
                for k_dilate in range(lim_k_dilate[0], lim_k_dilate[1], lim_k_dilate[2]):
                    for iterat_erode in range(lim_iterat_erode[0], lim_iterat_erode[1], lim_iterat_erode[2]):
                        for iterat_dilate in range(lim_iterat_dilate[0], lim_iterat_dilate[1], lim_iterat_dilate[2]):
                            for ratio_limit in np.arange(lim_ratio_limit[0], lim_ratio_limit[1], lim_ratio_limit[2]):
                                for white_threshold in range(lim_white_threshold[0], lim_white_threshold[1], lim_white_threshold[2]):
                                    for white_pixel_ratio_threshold in np.arange(lim_white_pixel_ratio_threshold[0], lim_white_pixel_ratio_threshold[1], lim_white_pixel_ratio_threshold[2]):
                                        i += 1
                                        print(i/(num_combinations))
                                    
                                        combinations.append([h_threshold, s_threshold, v_threshold, k_erode, k_dilate, iterat_erode, iterat_dilate, ratio_limit, white_threshold, white_pixel_ratio_threshold])
                                        puntuation_of_comb = []

                                        for image, dict_of_img in images_info_dict.items():
                                            image_path = 'selected_images/' + str(image)

                                            x_real = dict_of_img['x_real']
                                            y_real = dict_of_img['y_real']
                                            w_box = dict_of_img['w_box']
                                            h_box = dict_of_img['h_box']


                                            x_calc, y_calc, r_calc = compute_center(image_path, h_threshold, s_threshold, v_threshold, k_erode, k_dilate, iterat_erode, iterat_dilate, ratio_limit, white_threshold, white_pixel_ratio_threshold)

                                            correct_x = x_calc < x_real + w_box/2 and x_calc > x_real - w_box/2
                                            correct_y = y_calc < y_real + h_box/2 and y_calc > y_real - h_box/2

                                            if correct_x and correct_y:
                                                puntuation_of_comb.append(1)
                                            else:
                                                puntuation_of_comb.append(0)
                                            
                                            puntuations.append(puntuation_of_comb)



print('Puntuation: ', np.sum(puntuations))

#Save vectors
np.savetxt("combinations.txt", combinations, fmt='%d', delimiter='\t')
np.savetxt("puntuations.txt", puntuations, fmt='%d', delimiter='\t')

#Plot
def barplot(parameter, lim_parameter, special = False):
    name_params = ['h_threshold', 's_threshold', 'v_threshold', 'k_erode', 'k_dilate', 'iterat_erode', 'iterat_dilate', 'ratio_limit', 'white_threshold', 'white_pixel_ratio_threshold']
    index_param = name_params.index(parameter)
    categories=[]
    values = []

    if special:
        rang = np.arange(lim_parameter[0], lim_parameter[1], lim_parameter[2])
    else:
        rang = range(lim_parameter[0], lim_parameter[1], lim_parameter[2])

    for param in rang:
        freq = 0
        categories.append(str(param))
        score = 0
        
        for i, _ in enumerate(combinations): #len of combinations, which is the same len as puntuations
            combination = combinations[i]
            if combination[index_param] == param:
                freq += len(dataset)
                score += int(np.sum(puntuations[i]))          
        
        print(parameter + ': ' + str(param) + ' - Score: ' + str(score) + ' out of ' + str(freq))
        values.append(score)


    plt.bar(categories, values, color='salmon')
    plt.title(parameter + ' over ' + str(freq))
    plt.ylabel('Correct Prediction Frequency')
    plt.savefig(parameter + '.png')


barplot('h_threshold', lim_h_threshold)
barplot('s_threshold', lim_s_threshold)
barplot('v_threshold', lim_v_threshold)
barplot('k_erode', lim_k_erode)
barplot('k_dilate', lim_k_dilate)
barplot('iterat_erode', lim_iterat_erode)
barplot('iterat_dilate', lim_iterat_dilate)
barplot('ratio_limit', lim_ratio_limit, special = True)
barplot('white_threshold', lim_white_threshold)
barplot('white_pixel_ratio_threshold', lim_white_pixel_ratio_threshold, special = True) 