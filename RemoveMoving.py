from termcolor import colored
import numpy as np
import cv2
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from scipy import ndimage, misc
from skimage.metrics import structural_similarity as ssim
from skimage import filters
import time
import progressbar

def spot_the_diff(image1_gray, image2_gray, log=False, show_work=True, debug_init=True):
    ''' Given two images return the one with the object  
          image1_gray: [prev_bound - next_bound] * prev_show
          image2_gray: [prev_bound - next_bound] * next_frame
    ret=> if bound was correct the diff function will find dirt in image1 (=>1) or image1 and image2 are equals (=>0)
          if bound was incorrect the diff function will contain some moving parts of image2 (=>2)
             (can happen that also identifies small part like light changes of image1 but returns 2) ''' 

    if debug_init:
        global index
        plt.figure(figsize=(15,15))
        plt.subplot(121); plt.title("PREV {}".format(index))
        plt.imshow(cv2.cvtColor(image1_gray, cv2.COLOR_BGR2RGB))
        plt.subplot(122); plt.title("NEXT")
        plt.imshow(cv2.cvtColor(image2_gray, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.imsave("spot_debug/prev_{}.jpg".format(index),cv2.cvtColor(image1_gray, cv2.COLOR_BGR2RGB))
        plt.imsave("spot_debug/next_{}.jpg".format(index),cv2.cvtColor(image2_gray, cv2.COLOR_BGR2RGB))

    (score, sim) = ssim(image1_gray, image2_gray, full=True)
    sim = np.uint8(sim*255)
    thresh_sim =  np.uint8(np.logical_not(cv2.threshold(sim, 205, 255, cv2.THRESH_BINARY)[1])*255)
    thresh_sim = cv2.medianBlur(thresh_sim, 7)
    sum_sim = np.sum(thresh_sim>0)

      # lower bound (low thresh, high detection)
    if sum_sim<100:
        print(colored("Equality_l", 'blue'))
        return 0

    dif = cv2.absdiff(image1_gray, image2_gray)
    thresh_dif = cv2.medianBlur(dif, 5)

      # higher bound (high thresh, low detection)
    if not np.any(thresh_dif>50):
        print(colored("Equality_h", 'blue'))
        return 0

    if show_work:
        plt.figure(figsize=(20,20))
        plt.subplot(131); plt.title("ssim")
        plt.imshow(cv2.cvtColor(np.uint8(thresh_sim), cv2.COLOR_BGR2RGB))

    adaptive=0
    detected = 110
    a=None
    while(detected>80 if detected<100 else detected>105):
        a = np.uint8(thresh_dif>adaptive)*255
        detected = ((np.sum(a>0)/sum_sim)*100).round(3)
        if log: print("Not {}%".format(detected.round(2)))
        adaptive+=10

    mask = cv2.morphologyEx(a,cv2.MORPH_CLOSE, np.ones((12,12),np.uint8))
    mask = cv2.medianBlur(mask,3)
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((6,6),np.uint8))

    if show_work:
        plt.subplot(132); plt.title("Mask closed [{}]".format(adaptive-10))
        plt.imshow(cv2.cvtColor(np.uint8(mask), cv2.COLOR_BGR2RGB))
        plt.subplot(133); plt.title("Mask gradient [{}%]".format(detected.round(1)))
        plt.imshow(cv2.cvtColor(np.uint8(gradient), cv2.COLOR_BGR2RGB))
        plt.show()

      # canny in the border of the movement areas
    blurred1 = enhance_borders(image1_gray*(gradient>0))
    blurred2 = enhance_borders(image2_gray*(gradient>0))

    edges1 = auto_canny(blurred1)
    edges2 = auto_canny(blurred2)

      # count of canny oriented pixels in the border of the movement areas 
    i1=count_oriented(edges1)
    i2=count_oriented(edges2)
    if log: print(i1)
    if log: print(i2)

    if show_work:
        plt.figure(figsize=(20,20))
        plt.subplot(121)
        plt.title("Canny 1 [{}]".format(i1))
        plt.imshow(cv2.cvtColor(edges1, cv2.COLOR_BGR2RGB))
        plt.subplot(122)
        plt.title("Canny 2 [{}]".format(i2))
        plt.imshow(cv2.cvtColor(edges2, cv2.COLOR_BGR2RGB))
        plt.show()

    if show_work:
        plt.figure(figsize=(20,20))
        plt.subplot(121); plt.title("index {} sim_score {}".format(index, (score*100).round(1), 15), fontsize=20); 
        plt.imshow(cv2.cvtColor(image1_gray*(mask>0), cv2.COLOR_BGR2RGB))
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(image2_gray*(mask>0), cv2.COLOR_BGR2RGB))

      # prioritize the false negative cases to minimize the false positives 
    if i1>(i2+(0.01*i2)):
        if show_work: plt.title("Oggetto in 1", fontsize=20); plt.show()
        print (colored("#Oggetto in 1 [{}]".format(i1-i2),'blue')); return 1;
    else:
        if show_work: plt.title("Oggetto in 2", fontsize=20); plt.show()
        print(colored("#Oggetto in 2 [{}]".format(i2-i1),'blue')); return 2;


def expand_borders(m): return cv2.dilate(m, None, iterations=5)
def enhance_borders(im, k=np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])): return cv2.GaussianBlur(im, (3, 3), 0) # cv2.filter2D(im, -1, k) #
def count_oriented(img, show=False):
  ''' Count the points in a binary image taking in consideration the orientation '''

    # sobel derivatives
  derivX = cv2.Sobel(img, cv2.CV_32F, 1, 0)
  derivY = cv2.Sobel(img, cv2.CV_32F, 0, 1)

  mag = cv2.magnitude(np.absolute(derivX), np.absolute(derivY)) # absolute since a limitation of opencv from docs
  
    # thresholding of the magnitude values
  thresh = 1000
  _, mask = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)
  mask = np.uint8(mask>0)
  
  if show:
    plt.figure(figsize=(15,10)); plt.title("Mask magnitude min: {} max: {}".format(np.int(np.min(mag[mag>0])), np.int(np.max(mag))))
    plt.imshow(cv2.cvtColor(mask*255, cv2.COLOR_BGR2RGB))
    plt.show()
  
  return np.sum(mask)

def auto_canny(image, sigma=0.33):
	''' Apply Canny edge detector and thresholding using a lower and upper boundary on the gradient values.
	Lower value of sigma indicates a tighter threshold, whereas a larger value of sigma gives a wider threshold '''

	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

index=0
if __name__ == '__main__':
    path = '498-498.MOV'
    cap = cv2.VideoCapture(filename=path, apiPreference=cv2.CAP_FFMPEG)
    frame_numbers = [30, 45]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0]-1)
    ret, frame_1 = cap.read()
    image1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[1]-1)
    ret, frame_2 = cap.read()
    image2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    res = spot_the_diff(image1_gray, image2_gray, log=True,
                        show_work=True, debug_init=True)
    print(res)
    