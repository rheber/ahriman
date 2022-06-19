#!/usr/bin/env python

"""
Main file.
Much of this is based on https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
"""

import os
import joblib
from skimage.color import rgba2rgb, rgb2gray
from skimage.feature import hog
from skimage.io import imread as imgRead
from skimage.transform import resize as imgResize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Dimensions of the output files.
heightPx = 300
widthPx = 200
pklTitle = "cards"
pklname = f"output/{pklTitle}_{widthPx}x{heightPx}px.pkl"

def normalizeImages(srcDirs, includeFunc):
    """
    Load images, strip alpha channel, resize them and write them into a dict.

    The dict is written to a file named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    srcDirs: str
        paths to data
    include: (str)-> bool
        predicate for whether a file should be included
    """

    data = {} # The dict to be written.
    data['desc'] = f'resized {widthPx}x{heightPx}'
    data['label'] = []      # n class/folder names
    data['filename'] = []   # n file names
    data['data'] = []       # n images

    for folderPath in srcDirs: # For each class
        for filePath in os.listdir(folderPath): # For each instance
            if includeFunc(filePath): # If we should include the instance
                _fullPath = os.path.join(folderPath, filePath)
                #print(fullPath)
                im = imgRead(os.path.join(folderPath, filePath))
                if im.shape[2] == 4:
                    #print("removing alpha")
                    im = rgba2rgb(im)
                im = imgResize(im, (widthPx, heightPx))
                data['label'].append(folderPath)
                data['filename'].append(filePath)
                data['data'].append(im)
    joblib.dump(data, pklname)
    print('Completed resizing.')

def summary(data):
    """Print summary of the pickle file."""

    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['desc'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))

def displayExamples(data):
    """Display some of the pickled images."""

    # use np.unique to get all unique values in the list of labels
    labels = np.unique(data['label'])

    # set up the matplotlib figure and axes, based on the number of labels
    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15,4)
    fig.tight_layout()

    # make a plot for every label (equipment) type. The index method returns the
    # index of the first item corresponding to its search string, label in this case
    for ax, label in zip(axes, labels):
        idx = data['label'].index(label)
        ax.imshow(data['data'][idx])
        ax.axis('off')
        ax.set_title(label)
    plt.show()

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        pass

    def fit(self, _X, _y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([rgb2gray(img) for img in X])

class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

def split(data):
    """Split the given data into training and test data."""

    X = np.array(data['data'])
    y = np.array(data['label'])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )
    return (X_train, X_test, y_train, y_test)

grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(14, 14),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

def train(X_train, y_train):
    """Train a classifier on the given training data."""

    X_train_gray = grayify.fit_transform(X_train)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)

    classifier = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    classifier.fit(X_train_prepared, y_train)
    return classifier

def test(classifier, X_test, y_test):
    """Test the given classifier against the given test data."""

    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)

    y_pred = classifier.predict(X_test_prepared)
    print(np.array(y_pred == y_test)[:25])
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

# Preprocess
#normalizeImages(['../tcgeek/pokemon', '../tcgeek/yugioh'], lambda s: True)
pklData = joblib.load(pklname)

# Summarize
#summary(pklData)
#displayExamples(pklData)

# Train and test
(xtr, xts, ytr, yts) = split(pklData)
clsf = train(xtr, ytr)
#test(clsf, xts, yts)

# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

def drawBorder(image, imageWidth, imageHeight):
    """Draw a border and title onto the given image."""
    margin = 10
    borderWidth = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1

    cv2.line(image,(margin,margin),(imageWidth - margin,margin),(255,0,0),borderWidth)
    cv2.line(image,(margin,margin),(margin,imageHeight - margin),(255,0,0),borderWidth)
    cv2.line(image,(imageWidth - margin,margin),
             (imageWidth - margin,imageHeight - margin),(255,0,0),borderWidth)
    cv2.line(image,(margin,imageHeight - margin),
             (imageWidth - margin,imageHeight - margin),(255,0,0),borderWidth)
    cv2.putText(image,'Ahriman',(20,imageHeight - 20), font, fontScale,(255,255,255),2,cv2.LINE_AA)


# Display camera feed
cap = cv2.VideoCapture(0)
frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))
print(f"frame width: {frameWidth}, frame height: {frameHeight}")

defaultThreshold = 30
cannyThreshold = defaultThreshold
maxThreshold = 255
def setThreshold(val):
    """Callback to set the Canny threshold."""
    global cannyThreshold
    cannyThreshold = val

defaultBlur = 5
blur = defaultBlur
maxBlur = 100
def setBlur(val):
    """Callback to set the blur."""
    global blur
    blur = max(val, 1)

while True:
    #Capture each frame
    _, frame = cap.read()

    # https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (blur, blur))
    edges = cv2.Canny(blurred, cannyThreshold, cannyThreshold * 2)

    # https://stackoverflow.com/questions/37942132/opencv-detect-quadrilateral-in-python
    (cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    # loop over our contours
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(frame, [approx], -1, (0,255,0), 3)

    #drawBorder(frame, frameWidth, frameHeight)
    mainWindow = 'mainWindow'
    cv2.namedWindow(mainWindow)
    #cv2.createTrackbar('Canny Thresh:', mainWindow, cannyThreshold, maxThreshold, setThreshold)
    #cv2.createTrackbar('Blur:', mainWindow, blur, maxBlur, setBlur)
    cv2.imshow(mainWindow, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
