#!/usr/bin/env python

"""
Main file.
"""

import joblib
from skimage.io import imread as imgRead
from skimage.transform import resize as imgResize
import os

def resizeAll(srcDirs, pklname, includeFunc):
    """
    Based on https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
    Load images from path, resize them and write them as arrays to a dictionary,
    together with labels and metadata. The dictionary is written to a pickle file
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    srcDirs: str
        paths to data
    pklname: str
        path to output file
    include: (str)-> bool
        predicate for whether a file should be included
    """

    # Dimensions of the output files.
    heightPx = 300
    widthPx = 200

    data = {}
    data['desc'] = f'resized {widthPx}x{heightPx}'
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    pklname = f"output/{pklname}_{widthPx}x{heightPx}px.pkl"

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

resizeAll(['../tcgeek/pokemon', '../tcgeek/yugioh'], 'cards', lambda s: True)
