import os
import numpy as np
import argparse
import fastai
import scipy

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *

import ml_file_utils
import lambda_function

DELIM = ","
QUOTECHAR = '|'

def showResults(f2Filtered, dx):
    fig=plt.figure(figsize=(20, 20))
    fig.add_subplot(1,2, 1)
    plt.imshow(dx)
    plt.imshow(f2Filtered, alpha=0.9, cmap='hot');
    plt.show()


def setup():
    mlPath = os.environ['ML_PATH']+"/ml_data/ablob/"
    print("Path: {}".format(mlPath))
    #PATH = "machine_learning/ml_data/ablob/"
    sz = 224
    arch = resnet34
    bs = 64    

    m = arch(True)
    m = nn.Sequential(*children(m)[:-2], 
                      nn.Conv2d(512, 2, 3, padding=1), 
                      nn.AdaptiveAvgPool2d(1), Flatten(), 
                      nn.LogSoftmax())

    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_paths(mlPath, tfms=tfms, bs=bs)
    learn = ConvLearner.from_model_data(m, data)
    learn.load("trained_model_clipped_lobster")

    return m, learn, tfms, data

def loadData(imgName, targetPath, tfms, learn):
    
    ds = FilesIndexArrayDataset([imgName], np.array([0]), tfms[1], targetPath)
    dl = DataLoader(ds)
    preds = learn.predict_dl(dl)
    preds = np.argmax(preds)
    
    return dl, preds

def getMinMax(f2Filtered):
    minX = 1000
    minY = 1000
    maxX = 0
    maxY = 0

    for i in range(0,len(f2Filtered)):
        for j in range(0,len(f2Filtered[0])):
            val = f2Filtered[i][j]
            if val > 0:
                if i < minX:
                    minX = i
                if j < minY:
                    minY = j
                if i > maxX:
                    maxX = i
                if j > maxY:
                    maxY = j
    xRange = (minX, maxX)
    yRange = (minY, maxY)
    print("x range: ", xRange)
    print("y range: ", yRange)
    return xRange, yRange

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

def writeResults(xRange, yRange, filename, maxWidth):
    out_file = "ml_output.csv"
    maxWidth = 224;

    with open(out_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=DELIM, quotechar=QUOTECHAR, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Filename", "XStart", "XEnd", "YStart", "YEnd", "MaxWidth"])

        row = [filename, xRange[0], xRange[1], yRange[0], yRange[1], maxWidth]
        writer.writerow(row)


def read_args():
    ap = argparse.ArgumentParser()
    try:
        ap.add_argument("--image", required=True,
            help="path to the input image")
        ap.add_argument("--ref_object", required=True,
            help="")
        ap.add_argument("--ref_object_units", required=True,
            help="")
        ap.add_argument("--ref_object_size", required=True,
            help="")
        ap.add_argument("--fishery_type", required=True,
            help="")
        ap.add_argument("--uuid", required=True,
            help="")
        ap.add_argument("--username", required=True,
            help="")
        ap.add_argument("--email", required=True,
            help="")
        ap.add_argument("--original_filename", required=True,
            help="")
        ap.add_argument("--original_size", required=True,
            help="")
        ap.add_argument("--loc_code", required=True,
            help="")

        args = vars(ap.parse_args())
        print("reg args are ok...: ", args)
    except SystemExit as e:
        print(" regular args are wrong, error parsing args -> {}".format(e))
        ap = argparse.ArgumentParser()
        args = ap.parse_known_args()[1]

        print(".............................known args ->>> {}".format(args))
        parsedArgs = {}
        for dex, arg in enumerate(args):
            if str(arg).startswith('--'):
                strippedArg = arg.replace("--","").strip()
                parsedArgs[strippedArg] = args[dex+1]
        print("parsedArgs: ", parsedArgs)
        args = parsedArgs



    print("args -> {}".format(args))
    imageName = args["image"]
    ref_object = args["ref_object"]
    ref_object_units = args["ref_object_units"]
    ref_object_size = args["ref_object_size"]
    fishery_type = args["fishery_type"]
    uuid = args["uuid"]
    username = args["username"]
    email = args["email"]
    original_filename = args["original_filename"]
    original_size = args["original_size"]
    
    loc_code = args["loc_code"]

    return imageName, ref_object, ref_object_units, ref_object_size, fishery_type, uuid, username, email, original_filename, original_size, loc_code

def execute():
    print("here....")
    m, learn, tfms, data = setup();

    imageName, ref_object, ref_object_units, ref_object_size, fishery_type, uuid, username, email, original_filename, original_size, locCode = read_args()
    targetPath, imgName = os.path.split(imageName)
    print("target path: ", targetPath);

    print("imgName: ", imgName);
    dl, predictions = loadData(imgName, targetPath, tfms, learn)
    print("predictions--->>>>", predictions)
    x,y = next(iter(dl))
    x = x[None,0]
    vx = Variable(x.cpu(), requires_grad=True)
    dx = data.val_ds.denorm(x)[0]
    sfs = [SaveFeatures(o) for o in [m[-7], m[-6], m[-5], m[-4]]]

    py = m(Variable(x.cpu()))

    for o in sfs: o.remove()

    [o.features.size() for o in sfs]

    py = np.exp(to_np(py)[0]); py
    feat = np.maximum(0,to_np(sfs[3].features[0]))
    feat.shape

    
    f2=np.dot(np.rollaxis(feat,0,3), py)
    maxVal = f2.max()
    f2-=f2.min()
    f2/=f2.max()
    
    
    multiplier = 0.92
    if(predictions == 0):
        multiplier = 0.55

    #f2F = np.ma.masked_where(f2 <= 0.50, f2)
    filter = scipy.misc.imresize(f2, dx.shape,mode="L")
    maxVal = filter.max()*multiplier
    print("maxVal is {}".format(maxVal))

    f2Filtered = np.ma.masked_where(filter <=maxVal, filter)
    zeroMask = np.ma.filled(f2Filtered, 0)
    zeroMask[zeroMask > 0] = 255
    maskPath = os.environ['ML_PATH']+"/masks/"
    outMaskName = maskPath+imgName
    scipy.misc.imsave(outMaskName, zeroMask)


    #imageName, username, email, uuid, ref_object, ref_object_units, ref_object_size, locCode, fishery_type, original_filename, original_size
    jsonVals = lambda_function.runFromML(imageName, outMaskName, username, email, uuid, ref_object, ref_object_units, ref_object_size,
        locCode, fishery_type, original_filename, original_size)
    
    print("results---->>>>>>> ")
    print(jsonVals)



execute()
