import os
import numpy as np
import argparse
import scipy

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
import lambda_function
import uploads

DELIM = ","
QUOTECHAR = '|'
QUARTER_MODEL = "quarter_square_model_11_19"
AB_MODEL = "abalone_lobster_model_12_10"
SCALLOP_MODEL = "scallop_lobster_model_11_20"
AB_FULL_MODEL = "full_abalone_lobster_model_12_11"
FINFISH_MODEL = "finfish_multi_320_model_3_27_19a"

def showResults(f2Filtered, dx):
    fig=plt.figure(figsize=(20, 20))
    fig.add_subplot(1,2, 1)
    plt.imshow(dx)
    plt.imshow(f2Filtered, alpha=0.2, cmap='hot');
    plt.show()


def setup(fishery_type, loadFull=False):
    maxZoom=1.1
    tform = transforms_side_on
    print("--->>>fishery type: {}".format(fishery_type))
    numTypes = 2

    if "scallop" in fishery_type:
        print("looking for scallops...")
        mlPath = os.environ['ML_PATH']+"/ml_data/scallop/"
        sz = 320
        model_name = SCALLOP_MODEL
        maxZoom = 1.1
    elif "lobster" in fishery_type and loadFull:
        mlPath = os.environ['ML_PATH']+"/ml_data/ablob_full/"
        sz = 320
        model_name = AB_FULL_MODEL
        maxZoom = 1.1
    elif "finfish" in fishery_type:
        numTypes = 4
        print("--->>>>>finfish")
        mlPath = os.environ['ML_PATH']+"/ml_data/finfish_multi/"
        sz = 320
        model_name = FINFISH_MODEL
        maxZoom = 1.0
    else:    
        print('--->>>looking for abalone')
        mlPath = os.environ['ML_PATH']+"/ml_data/ablob/"
        tform = transforms_top_down
        sz = 320
        model_name = AB_MODEL
        maxZoom = 1.1

    #PATH = "machine_learning/ml_data/ablob/"
    
    arch = resnet34
    bs = 64    

    m = arch(True)
    #stride (second arg) needs to match num classes, otherwise assert is thrown in pytorch
    m = nn.Sequential(*children(m)[:-2], 
                      nn.Conv2d(512, numTypes, 3, padding=1,groups=1), 
                      nn.AdaptiveAvgPool2d(1), Flatten(), 
                      nn.LogSoftmax())


    tfms = tfms_from_model(arch, sz, aug_tfms=tform, max_zoom=maxZoom)
    data = ImageClassifierData.from_paths(mlPath, tfms=tfms, bs=bs)
    learn = ConvLearner.from_model_data(m, data)
    learn.load(model_name)

    return m, tfms, data,learn

def setupQuarterSquareModel():
    mlPath = os.environ['ML_PATH']+"/ml_data/quarter_square/"
    #PATH = "machine_learning/ml_data/ablob/"
    sz = 480
    arch = resnet34
    bs = 64    

    m = arch(True)
    #stride (second arg) needs to match num classes, otherwise assert is thrown in pytorch
    m = nn.Sequential(*children(m)[:-2], 
                      nn.Conv2d(512, 2, 3, padding=1), 
                      nn.AdaptiveAvgPool2d(1), Flatten(), 
                      nn.LogSoftmax())


    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_paths(mlPath, tfms=tfms, bs=bs)
    learn = ConvLearner.from_model_data(m, data)
    learn.load(QUARTER_MODEL)


    return m, tfms, data,learn

def loadData(imgName, targetPath, tfms, learn):
    ds = FilesIndexArrayDataset([imgName], np.array([0]), tfms[1], targetPath)
    dl = DataLoader(ds)
    preds = learn.predict_dl(dl)
    preds = np.argmax(preds)
    
    return dl, preds

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

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
        ap.add_argument("--show", required=False,
            help="")

        args = vars(ap.parse_args())
    except SystemExit as e:
        ap = argparse.ArgumentParser()
        args = ap.parse_known_args()[1]

        parsedArgs = {}
        for dex, arg in enumerate(args):
            if str(arg).startswith('--'):
                strippedArg = arg.replace("--","").strip()
                parsedArgs[strippedArg] = args[dex+1]

        args = parsedArgs

    imageName = args["image"]
    try:
        showResults = args["show"]
    except KeyError as ke:
        showResults=False

    print("args-->>{}".format(args))
    try:
        hasRefObject = True
        ref_object = args["ref_object"]
        print("ref object: {}".format(ref_object))
    except KeyError as ke:
        print("key err: {}".format(ke))
        hasRefObject = False

    
    if not hasRefObject:
        #for running it locally
        print(" batch -- falling back to abalone & quarter")
        ref_object = "quarter"
        ref_object_units = "inches"
        ref_object_size = 0.955
        fishery_type = "finfish"
        uuid = str(time.time()*1000)
        username = "dytest"
        email = "none given"
        original_filename = "none given"
        original_size = None
        loc_code = "Fake Place"
    else:

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
        showResults = args["show"]
    
    print("show: {}".format(showResults))
    return imageName, ref_object, ref_object_units, ref_object_size, fishery_type, uuid, username, email, original_filename, original_size, loc_code, showResults

def runModel(m, tfms, data, learn, imgName, targetPath, multiplier, restrictedMultiplier, show, extraMask, isQuarterOrSquare,fullPrefix=""):
    dl, predictions = loadData(imgName, targetPath, tfms, learn)

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
    
    
    f2=np.dot(np.rollaxis(feat,0,3), py)
    maxVal = f2.max()
    f2-=f2.min()
    f2/=f2.max()

    #f2F = np.ma.masked_where(f2 <= 0.50, f2)
    filter = scipy.misc.imresize(f2, dx.shape,mode="L")
    maxVal = filter.max()*multiplier
    
    rMaxVal = filter.max()*restrictedMultiplier

    f2Filtered = np.ma.masked_where(filter <=maxVal, filter)
    zeroMask = np.ma.filled(f2Filtered, 0)
    zeroMask[zeroMask > 0] = 255

    rZeroMask = None
    if restrictedMultiplier > 0:
        print("writing restriction mask for quarters/squares")

        rF2Filtered = np.ma.masked_where(filter <= rMaxVal, filter)
        rZeroMask = np.ma.filled(rF2Filtered, 0)
        rZeroMask[rZeroMask > 0] = 255

    #do extra masking step of target area
    #if extraMask is not None:
    if False:
        #zeroMask = np.add(zeroMask, extraMask)
        zeroMask[extraMask == 255] = 0

    if isQuarterOrSquare:
        maskPath = os.environ['ML_PATH']+"/qs_masks/"
    else:
        maskPath = os.environ['ML_PATH']+"/masks/"

    outMaskName = maskPath+fullPrefix+imgName
    if not os.path.isdir(maskPath):
        os.mkdir(maskPath)

    writeMask(zeroMask, outMaskName, False)

    if show:
        showResults(zeroMask, dx)

    return rZeroMask, outMaskName

def writeMask(zeroMask, outMaskName, show=False):
    #make sure doesn't hit edges
    nrows, ncols = zeroMask.shape
    
    square_mask = np.ones((nrows, ncols), dtype=bool)
    for (x,y), value in np.ndenumerate(square_mask):
        square_mask[x][y] = x<2 or x > nrows-2 or y<2 or y>ncols-2

    #zeroMask[outer_edge_mask] = 0
    zeroMask[square_mask] = 0
    cv2.imwrite(outMaskName, zeroMask)

    if False:
        cv2.imshow("mask ", zeroMask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def isLobster(fishery_type):
    return "lobster" in fishery_type

def isScallop(fishery_type):
    return "scallop" in fishery_type

def isFinfish(fishery_type):
    return "finfish" in fishery_type

def isAbalone(fishery_type):
    return "abalone" in fishery_type

def execute():
    imageName, ref_object, ref_object_units, ref_object_size, fishery_type, uuid, username, email, original_filename, original_size, locCode, showResults = read_args()
    m, tfms, data, learn = setup(fishery_type);
    print("fishery type: {}".format(fishery_type))

    if fishery_type == "square_test":
        jsonVals = lambda_function.runFromML(imageName, None, None, username, email, uuid, ref_object, ref_object_units, ref_object_size,
            locCode, fishery_type, original_filename, original_size, "", showResults)
        print(">>>>><<<<<")
        print(jsonVals)
        return jsonVals
    else:
        if isLobster(fishery_type):
            print("setting up full lobster...")
            fullM, fullTfms, fullData, fullLearn = setup(fishery_type, True)

        quarterSquareModel, qsTfms, qsData, qsLearn = setupQuarterSquareModel()

        targetPath, imgName = os.path.split(imageName)
        
        if imageName == None:
            return

        multiplier = 0.85
        rMultiplier = 0.85
        if(isAbalone(fishery_type)):
            multiplier = 0.40
            rMultiplier = 0.5
        elif(isScallop(fishery_type)):
            multiplier = 0.30
            rMultiplier = 0.5
        elif(isFinfish(fishery_type)):
            multiplier = 0.36
            rMultiplier = 0.5

        tmpImgName = None
        print("running model for ablob...: {}".format(fishery_type))
        print("multiplier: {}".format(multiplier))

        if(isLobster(fishery_type)):
            print("doing clipped lobster")
            zeroMask, outMaskName = runModel(fullM, fullTfms, fullData, fullLearn, imgName, targetPath, 0.92, rMultiplier, False, None, False)
        else:
            zeroMask, outMaskName = runModel(m, tfms, data, learn, imgName, targetPath, multiplier, rMultiplier, True, None, False)

        fullMaskName = ""
        if isLobster(fishery_type):
            fullZeroMask, fullMaskName = runModel(fullM, fullTfms, fullData, fullLearn, imgName, targetPath, 0.35, rMultiplier, False, None, False, "full_")
        
        if ref_object == "square":
            extraMask = None
        else:
            extraMask = zeroMask
        
        extraMask, extraMaskName = runModel(quarterSquareModel, qsTfms, qsData, qsLearn, imgName, targetPath, 0.5, 0, False, extraMask, True)
        print("done with ml")
        #imageName, username, email, uuid, ref_object, ref_object_units, ref_object_size, locCode, fishery_type, original_filename, original_size
        jsonVals = lambda_function.runFromML(imageName, outMaskName, fullMaskName, username, email, uuid, ref_object, ref_object_units, ref_object_size,
            locCode, fishery_type, original_filename, original_size, extraMaskName, showResults)
        print(">>>>><<<<<")
        print(jsonVals)
        return jsonVals


execute()
