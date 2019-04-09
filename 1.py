# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np

def nothing(x):
    pass


def doDenoise(img, faktor):
    # Faktör Sıfıra yaklaştıkça daha gürültü artar. Varsayılan: 1
    # Kaynak: http: // docs.opencv.org / trunk / d5 / d69 / tutorial_py_non_local_means.html
    img = cv2.fastNlMeansDenoising(img, None, faktor, 21, 7)
    return img

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,20,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

cap = cv2.VideoCapture(0)

i = 0
while(True):

    start_time = time.time()
    i+=1

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')


    ret, frame = cap.read()
    # Our operations on the frame come here
    if r>0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = doDenoise(frame, r)
    # Display the resulting frame
    cv2.imshow('image', frame)

    elapsed_time = time.time() - start_time
    print(i, "----", elapsed_time)

    ## time.sleep( elapsed_time )

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break


    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()


exit()






import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

i = 0
while(True):

    start_time = time.time()
    i+=1

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    elapsed_time = time.time() - start_time
    print(i, "----", elapsed_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()

exit()

# returns OpenCV VideoCapture property id given, e.g., "FPS"
def capPropId(prop):
  return getattr(cv2 if OPCV3 else cv2.cv,
    ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)

from pkg_resources import parse_version
OPCV3 = parse_version(cv2.__version__) >= parse_version('3')

#set the width and height, and UNSUCCESSFULLY set the exposure time
cap.set(3,1280)
cap.set(4,1024)
cap.set(15, 0.1)



exit()

import matplotlib.pyplot as plt
import cv2

im = cv2.imread('../imgs/03.jpg')
# calculate mean value from RGB channels and flatten to 1D array
vals = im.mean(axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0, 255])
plt.show()

exit()
# draw histogram in python.
import cv2
import numpy as np






exit()

import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
#img0 = cv2.imread('SanFrancisco.jpg',)
img0 = cv2.imread("../imgs/03.jpg")

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3), 0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

exit()

row,col,ch = img.shape
p = 0.9
a = 0.1
noisy = img

# Salt mode
num_salt = np.ceil(a * img.size * p)
coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in img.shape]
noisy[coords] = 1

# Pepper mode
num_pepper = np.ceil(a * img.size * (1. - p))
coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in img.shape]
noisy[coords] = 0

cv2.imshow('noisy', noisy)

# Here is the code to use the median filter:

median_blur= cv2.medianBlur(img, 3)
cv2.imshow('median_blur', median_blur)

cv2.waitKey()
cv2.destroyAllWindows()


exit()

### Demoise Uygulaması,
### Demoise Uygulaması, Kaynak: http://docs.opencv.org/trunk/d5/d69/tutorial_py_non_local_means.html
### Demoise Uygulaması, Kaynak: http://docs.opencv.org/trunk/d5/d69/tutorial_py_non_local_means.html
import numpy as np
import cv2
from matplotlib import pyplot as plt


def doAdjustGamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def doGreyscale(img):
    # converting to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def doDenoise(img, faktor):
    # Faktör Sıfıra yaklaştıkça daha gürültü artar. Varsayılan: 1
    # Kaynak: http: // docs.opencv.org / trunk / d5 / d69 / tutorial_py_non_local_means.html
    img = cv2.fastNlMeansDenoising(img, None, faktor, 21, 7)
    return img

def doCanny(img, n = 100, m = 10):
    # Canny recommended a upper:lower ratio between 2:1 and 3:1.
    # http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Canny_Edge_Detection.php
    img = cv2.Canny(img, n, m)
    return img

def doMedianBlur(img, factor):
    # Factor, 1-9 arasıdır.
    img = cv2.medianBlur(img, factor)
    return img

def doGaussianBlur(img, factor):
    # remove noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def doHistogram2(gray_img):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    plt.hist(gray_img.ravel(), 256, [0, 256]);
    plt.show()
    return hist

def doHistogram3(img):
    #kaynak: https://pythonspot.com/en/image-histogram/
    h = np.zeros((300, 256, 3))

    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img], [ch], None, [256], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)

    h = np.flipud(h)
    cv2.imshow('Histogram', h)


# Load the image
img = cv2.imread("../imgs/03.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('ORJINAL',img)

faktor = 4
img = doDenoise(img, faktor)
cv2.imshow('Denoise ' + str(faktor),img)

img1 = doAdjustGamma(img, 0.5)
img2 = doAdjustGamma(img, 1.0)
img3 = doAdjustGamma(img, 1.5)

cv2.imshow("Gamma 0.5", img1)
cv2.imshow("Gamma 1.0", img2)
cv2.imshow("Gamma 1.5", img3)


cv2.imwrite("img1.jpg", img1)
cv2.imwrite("img2.jpg", img2)
cv2.imwrite("img3.jpg", img3)


n = 25
m = 1
img = doCanny(img, n, m)
cv2.imshow('Canny ' + str(faktor),img)


"""

# Loading exposure images into a list
fastNlMeansDenoisingColored = ["img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv2.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

# Merge exposures to HDR image
merge_debvec = cv2.createMergeDebevec()
#hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
merge_robertson = cv2.createMergeRobertson()
#hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())


# Tonemap HDR image
tonemap1 = cv2.createTonemapDurand(gamma=2.2)
#res_debvec = tonemap1.process(hdr_debvec.copy())
tonemap2 = cv2.createTonemapDurand(gamma=1.3)
#res_robertson = tonemap2.process(hdr_robertson.copy())


# Exposure fusion using Mertens
merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Convert datatype to 8-bit and save
#res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
#res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

#cv2.imwrite("aaaldr_debvec.jpg", res_debvec_8bit)
#cv2.imwrite("aaaldr_robertson.jpg", res_robertson_8bit)
cv2.imwrite("aaafusion_mertens.jpg", res_mertens_8bit)

#cv2.imshow("aaaldr_debvec.jpg", cv2.imread("aaaldr_debvec.jpg"))
#cv2.imshow("aaaldr_robertson.jpg", cv2.imread("aaaldr_robertson.jpg"))
cv2.imshow("aaafusion_mertens.jpg", cv2.imread("aaafusion_mertens.jpg"))

"""



# Finally show the image
#cv2.imshow('SONUC',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

