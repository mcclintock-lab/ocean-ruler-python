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
AB_MODEL = "abalone_lobster_model_12_5"
SCALLOP_MODEL = "scallop_lobster_model_11_20"

def showResults(f2Filtered, dx):
    fig=plt.figure(figsize=(20, 20))
    fig.add_subplot(1,2, 1)
    plt.imshow(dx)
    plt.imshow(f2Filtered, alpha=0.6, cmap='hot');
    plt.show()


def setup(fishery_type):
    if fishery_type == "scallop":
        print("looking for scallops...")
        mlPath = os.environ['ML_PATH']+"/ml_data/scallop/"
        sz = 320
        model_name = SCALLOP_MODEL
    else:    
        mlPath = os.environ['ML_PATH']+"/ml_data/ablob/"
        sz = 480
        model_name = AB_MODEL
    #PATH = "machine_learning/ml_data/ablob/"
    
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
    print("args-->>{}".format(args))
    try:
        hasRefObject = True
        ref_object = args["ref_object"]
        print("ref object: {}".format(ref_object))
    except KeyError as ke:
        print("key err: {}".format(ke))
        hasRefObject = False

    if not hasRefObject:
        print(" batch -- falling back to abalone & quarter")
        ref_object = "square"
        ref_object_units = "cm"
        ref_object_size = 5.0
        fishery_type = "scallop"
        uuid = str(time.time()*1000)
        username = "none given"
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
        print("ref object units::: {}".format(ref_object_units))
    
    return imageName, ref_object, ref_object_units, ref_object_size, fishery_type, uuid, username, email, original_filename, original_size, loc_code

def runModel(m, tfms, data, learn, imgName, targetPath, multiplier, restrictedMultiplier, show, extraMask, isQuarterOrSquare):
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
    feat.shape
    
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
        #rMaskPath = os.environ['ML_PATH']+"rmasks"    
        #if not os.path.isdir(rMaskPath):
        #    os.mkdir(rMaskPath)

        rF2Filtered = np.ma.masked_where(filter <= rMaxVal, filter)
        rZeroMask = np.ma.filled(rF2Filtered, 0)
        rZeroMask[rZeroMask > 0] = 255
        #outRMaskPath = rMaskPath+imgName
        #writeMask(rZeroMask, outRMaskPath, True)

    #do extra masking step of target area
    #if extraMask is not None:
    if False:
        #zeroMask = np.add(zeroMask, extraMask)
        zeroMask[extraMask == 255] = 0

    if isQuarterOrSquare:
        maskPath = os.environ['ML_PATH']+"/qs_masks/"
    else:
        maskPath = os.environ['ML_PATH']+"/masks/"

    outMaskName = maskPath+imgName
    if not os.path.isdir(maskPath):
        os.mkdir(maskPath)

    writeMask(zeroMask, outMaskName, False)

    if show:
        showResults(zeroMask, dx)

    return rZeroMask, outMaskName

def writeMask(zeroMask, outMaskName, show=False):
    #make sure doesn't hit edges
    nrows, ncols = zeroMask.shape
    row, col = np.ogrid[:nrows, :ncols]
    cnt_row, cnt_col = nrows / 2, ncols / 2
    outer_edge_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > ((nrows / 2)-4)**2 )
    zeroMask[outer_edge_mask] = 0
    cv2.imwrite(outMaskName, zeroMask)

    if show:
        cv2.imshow("mask ", zeroMask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def execute():
    imageName, ref_object, ref_object_units, ref_object_size, fishery_type, uuid, username, email, original_filename, original_size, locCode = read_args()
    m, tfms, data, learn = setup(fishery_type);
    #quarterSquareModel, qsTfms, qsData, qsLearn = setupQuarterSquareModel()

    targetPath, imgName = os.path.split(imageName)
    
    if imageName == None:
        return

    multiplier = 0.85
    rMultiplier = 0.85
    if(fishery_type == 'abalone'):
        multiplier = 0.40
        rMultiplier = 0.5
    elif(fishery_type == 'scallop'):
        multiplier = 0.30
        rMultiplier = 0.5

    tmpImgName = None
    extraMask, outMaskName = runModel(m, tfms, data, learn, imgName, targetPath, multiplier, rMultiplier, False, None, False)
    if ref_object == "square":
        extraMask = None

    #extraMask, outMaskName = runModel(quarterSquareModel, qsTfms, qsData, qsLearn, imgName, targetPath, 0.5, 0, True, extraMask, True)
    
    print("done with ml")
    #imageName, username, email, uuid, ref_object, ref_object_units, ref_object_size, locCode, fishery_type, original_filename, original_size
    jsonVals = lambda_function.runFromML(imageName, outMaskName, username, email, uuid, ref_object, ref_object_units, ref_object_size,
        locCode, fishery_type, original_filename, original_size)
    print(">>>>><<<<<")
    print(jsonVals)
    return jsonVals






execute()
