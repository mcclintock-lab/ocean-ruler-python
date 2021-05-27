import os
import numpy as np
import argparse
import scipy
import utils

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from PIL import Image
import lambda_function
import uploads

DELIM = ","
QUOTECHAR = '|'
QUARTER_MODEL = "quarter_square_model_11_19"
AB_MODEL = "abalone_lobster_model_12_10"
SCALLOP_MODEL = "scallop_lobster_model_11_20"
AB_FULL_MODEL = "full_abalone_lobster_model_12_11"
FINFISH_MODEL = "finfish_lobster_320_model_4_1_19"

def showResults(f2Filtered, dx):
    fig=plt.figure(figsize=(4, 4))
    fig.add_subplot(1,1, 1)
    plt.imshow(dx)
    plt.imshow(f2Filtered, alpha=0.2, cmap='hot');
    plt.show()


def setup(fishery_type, loadFull=False):
    """ Load the correct model based on tthe fishery type
        fishery_type is one of the options defined in constants.py
        loadFull is for lobsters only to get the entire image
        Note: there won't be models for new fisheries, so by default it falls back to
        a shellfish (abalone) model
    """
    maxZoom=1.1
    tform = transforms_side_on

    numTypes = 2
    if "scallop" in fishery_type:
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
        numTypes = 2
        mlPath = os.environ['ML_PATH']+"/ml_data/finfish_multi/"
        sz = 320
        model_name = FINFISH_MODEL
        maxZoom = 1.0
    else:
        mlPath = os.environ['ML_PATH']+"/ml_data/ablob/"
        tform = transforms_top_down
        sz = 320
        model_name = AB_MODEL
        maxZoom = 1.1


    #the fastai model loading and application
    arch = resnet34
    bs = 64

    m = arch(True)
    #stride (second arg) needs to match num classes, otherwise assert is thrown in pytorch
    m = nn.Sequential(*children(m)[:-2],
                      nn.Conv2d(512, numTypes, 3, padding=1,groups=1),
                      nn.AdaptiveAvgPool2d(1), Flatten(),
                      nn.LogSoftmax())

    #load the model from the transforms
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, aug_tfms=tform, max_zoom=maxZoom)
    data = ImageClassifierData.from_paths(mlPath, tfms=tfms, bs=bs)
    #and run the model on the new data
    learn = ConvLearner.from_model_data(m, data)
    learn.load(model_name)

    return m, tfms, data,learn


def setupQuarterSquareModel():
    """Set up the fastai model for the quarter or square (the reference object)

    """
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


    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_paths(mlPath, tfms=tfms, bs=bs)
    learn = ConvLearner.from_model_data(m, data)
    learn.load(QUARTER_MODEL)

    return m, tfms, data,learn


def loadData(imgName, targetPath, tfms, learn):
    """load image and run fastai predict"""
    ds = FilesIndexArrayDataset([imgName], np.array([0]), tfms[1], targetPath)
    dl = DataLoader(ds)

    preds = learn.predict_dl(dl)
    preds = np.argmax(preds)

    return dl

class SaveFeatures():
    """Hooks for inserting into the model so that the probability data is saved out"""
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
        ap.add_argument('--measurement_direction', required=True,help="")

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


    try:
        hasRefObject = True
        ref_object = args["ref_object"]
        print("ref object: {}".format(ref_object))
    except KeyError as ke:
        print("key err: {}".format(ke))
        hasRefObject = False


    if not hasRefObject:
        #for running it locally, not through the server. these are all deefaults
        print(" batch -- falling back to abalone & quarter")
        ref_object = "quarter"
        ref_object_units = "cm"
        ref_object_size = 2.426
        fishery_type = "california_finfish"
        uuid = str(time.time()*1000)
        username = "dytest"
        email = "none given"
        original_filename = "none given"
        original_size = None
        loc_code = "Fake Place"
        measurement_direction = "length"
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
        measurement_direction = args["measurement_direction"]

    print("show: {}".format(showResults))
    return imageName, ref_object, ref_object_units, ref_object_size, fishery_type, uuid, username, email, original_filename, original_size, loc_code, showResults, measurement_direction

def runModel(m, tfms, data, learn, imgName, targetPath, multiplier, restrictedMultiplier, show, extraMask, isQuarterOrSquare,fullPrefix=""):
    """ Run the fastai model for a fishery


    """
    dl = loadData(imgName, targetPath, tfms, learn)

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

    new_shape = (dx.shape[0],dx.shape[1])
    filter = np.array(Image.fromarray(f2).resize(new_shape))
    maxVal = filter.max()*multiplier

    rMaxVal = filter.max()*restrictedMultiplier

    f2Filtered = np.ma.masked_where(filter <=maxVal, filter)
    zeroMask = np.ma.filled(f2Filtered, 0)
    zeroMask[zeroMask > 0] = 255

    rZeroMask = None
    if restrictedMultiplier > 0:

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

'''
def get_orientation_info(imageName):
    print("image name: {}".format(imageName))
    img=Image.open(imageName)
    #for orientation in ExifTags.TAGS.keys():

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
    }
    #exif=dict(pil_image._getexif().items())
    print("EXIF DATA: {}".format(exif))
    return exif['orientation']
'''

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




def execute():
    """ Main look called from ocean-ruler server.
        It loads the model object and processes the data based on the fishery type and reference type specified

    """
    imageName, ref_object, ref_object_units, ref_object_size, fishery_type, uuid, username, email, original_filename, original_size, locCode, showResults, measurementDirection = read_args()
    m, tfms, data, learn = setup(fishery_type)

    if utils.isLobster(fishery_type):
        fullM, fullTfms, fullData, fullLearn = setup(fishery_type, True)

    quarterSquareModel, qsTfms, qsData, qsLearn = setupQuarterSquareModel()


    targetPath, imgName = os.path.split(imageName)

    if imageName == None:
        return

    #the multiplier is used to determine how much to 'trust' the model output
    #the higher the number, the more precise the model is expected to be, and the
    #output will be clipped more precisely
    #if ends of the input is getting chopped off (nose and tail of fish, e.g., drop the number)
    multiplier = 0.85
    rMultiplier = 0.85
    if(utils.isAbalone(fishery_type)):
        multiplier = 0.35
        rMultiplier = 0.5
    elif(utils.isScallop(fishery_type)):
        multiplier = 0.30
        rMultiplier = 0.5
    elif(utils.isFinfish(fishery_type)):
        #going higher than 0.32 starts to chop off fish edges...
        multiplier = 0.2
        rMultiplier = 0.5
    else:
        multiplier = 0.4
        rMultiplier = 0.5


    #lobster is so very special, run it differently
    if(utils.isLobster(fishery_type)):
        zeroMask, outMaskName = runModel(fullM, fullTfms, fullData, fullLearn, imgName, targetPath, 0.92, rMultiplier, False, None, False)
    else:
        zeroMask, outMaskName = runModel(m, tfms, data, learn, imgName, targetPath, multiplier, rMultiplier, False, None, False)

    fullMaskName = ""
    #lobster also has a mask of the whole image and the carapace
    if utils.isLobster(fishery_type):
        _, fullMaskName = runModel(fullM, fullTfms, fullData, fullLearn, imgName, targetPath, 0.45, rMultiplier, False, None, False, "full_")


    if ref_object == "square":
        refObjectMask = None
    else:
        refObjectMask = zeroMask

    if not utils.isQuarter(ref_object):
        if fishery_type == "mussels":
            #don't mask the mussels ref object - a lot of them are too close to the target so it breaks things...
            #this decreases accuracy, but allows it to work...
            refObjectMaskName = imageName
        else:
            refObjectMask, refObjectMaskName = runModel(quarterSquareModel, qsTfms, qsData, qsLearn, imgName, targetPath, 0.2, 0, False, refObjectMask, True)

    else:
        print("getting quarter masks...")
        #same with quarter. too many cases where its right next to the target (abalone)
        #refObjectMask, refObjectMaskName = runModel(quarterSquareModel, qsTfms, qsData, qsLearn, imgName, targetPath, 0.20, 0, False, refObjectMask, True)
        refObjectMaskName = imageName

    #run the computer visions steps after the machine learning is done
    jsonVals = lambda_function.runFromML(imageName, outMaskName, fullMaskName, username, email, uuid, ref_object, ref_object_units, ref_object_size,
        locCode, fishery_type, original_filename, original_size, refObjectMaskName, showResults, measurementDirection)
    #the web client is looking for this to split the resultant json from the cruft that AWS adds to the ends of the json
    #this may not be necessary now that API Gateway isn't being used, but I haven't had time to go check
    print(">>>>><<<<<")
    print(jsonVals)
    return jsonVals


execute()
