#!/usr/bin/env python

"""
Main file.
"""

import joblib
from skimage.io import imread as imgRead
from skimage.transform import resize as imgResize
import numpy as np
import matplotlib.pyplot as plt
import os

# Dimensions of the output files.
heightPx = 300
widthPx = 200
pklTitle = "cards"
pklname = f"output/{pklTitle}_{widthPx}x{heightPx}px.pkl"

def resizeAll(srcDirs, includeFunc):
    """
    Based on https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
    Load images from path, resize them and write them as arrays to a dictionary,
    together with labels and metadata. The dictionary is written to a pickle file
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    srcDirs: str
        paths to data
    include: (str)-> bool
        predicate for whether a file should be included
    """

    data = {}
    data['desc'] = f'resized {widthPx}x{heightPx}'
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    for folderPath in srcDirs:
        for filePath in os.listdir(folderPath):
            if includeFunc(filePath):
                fullPath = os.path.join(folderPath, filePath)
                print(fullPath)
                im = imgRead(os.path.join(folderPath, filePath))
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

#resizeAll(['../tcgeek/pokemon', '../tcgeek/yugioh'], lambda s: True)
pklData = joblib.load(pklname)
summary(pklData)
displayExamples(pklData)
