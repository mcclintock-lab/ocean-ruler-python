

import fastai
import scipy

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *

import ml_file_utils

DELIM = ","
QUOTECHAR = '|'

def showResults(f2Filtered, dx):
    fig=plt.figure(figsize=(20, 20))
    fig.add_subplot(1,2, 1)
    plt.imshow(dx)
    plt.imshow(f2Filtered, alpha=0.9, cmap='hot');
    plt.show()


def setup():
    PATH = "ml_data/ablob/"
    sz = 224
    arch = resnet34
    bs = 64    

    m = arch(True)
    m = nn.Sequential(*children(m)[:-2], 
                      nn.Conv2d(512, 2, 3, padding=1), 
                      nn.AdaptiveAvgPool2d(1), Flatten(), 
                      nn.LogSoftmax())

    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)
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

def execute():
    m, learn, tfms, data = setup();


    targetPath, imgName = ml_file_utils.read_args()
    
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
    scipy.misc.imsave("masks/"+imgName, zeroMask)


    #showResults(zeroMask, dx)

    maxWidth = 224;


execute()
