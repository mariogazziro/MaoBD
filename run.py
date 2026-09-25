import os
import contextlib
from MaoBD import MaoBD

import imhandle as imh

dataPath = './Images'
outPath = './Output'
resultsPath = './results.txt'

if not os.path.isdir(outPath):
    os.mkdir(outPath)

handPics = [os.path.basename(fn) for fn in imh.image_names_in_folder(dataPath)]

with open(resultsPath, 'wt', buffering=1) as results, contextlib.redirect_stdout(results):
    for i in range(0, len(handPics)):
        #print handPics[i]
        handSide = os.path.splitext(handPics[i])[0][-1].upper()
        if handSide=='D':
            maobd = MaoBD(dataPath, handPics[i], "RIGHT", outPath)
        else:
            if handSide=='E':
                maobd = MaoBD(dataPath, handPics[i], "LEFT", outPath)
