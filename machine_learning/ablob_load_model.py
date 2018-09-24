
# coding: utf-8

# In[1]:


from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
import cv2


# In[4]:


PATH = "ml_data/ablob/"
sz = 224
arch = resnet34
bs = 64


# In[3]:


m = arch(True)
m = nn.Sequential(*children(m)[:-2], 
                  nn.Conv2d(512, 2, 3, padding=1), 
                  nn.AdaptiveAvgPool2d(1), Flatten(), 
                  nn.LogSoftmax())
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)
learn = ConvLearner.from_model_data(m, data)
learn.load("trained_model")


# In[25]:


imgName = "GOPR1688.JPG"
targetPath = "ml_data/ablob/test/lobster/"
#imgName = "blue_abalone_rot.jpg"
#targetPath = "data/ablob/test/abalone/"
#imgName="lobster.jpg"
#targetPath="data/ablob/test/"
targetImagePath = targetPath+imgName
targetImage = Image.open(targetImagePath).resize((224,224))
plt.imshow(targetImage)


# In[26]:



ds = FilesIndexArrayDataset([imgName], np.array([0]), tfms[1], targetPath)
dl = DataLoader(ds)
preds = learn.predict_dl(dl)
np.argmax(preds)


# ## CAM

# In[27]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


# In[28]:


x,y = next(iter(dl))


# In[29]:


x = x[None,0]
vx = Variable(x.cpu(), requires_grad=True)


# In[30]:


dx = data.val_ds.denorm(x)[0]
#plt.imshow(dx);


# In[31]:


sfs = [SaveFeatures(o) for o in [m[-7], m[-6], m[-5], m[-4]]]


# In[32]:

py = m(Variable(x.cpu()))
#get_ipython().magic(u'time py = ')


# In[33]:


for o in sfs: o.remove()


# In[34]:


[o.features.size() for o in sfs]


# In[35]:


py = np.exp(to_np(py)[0]); py


# In[36]:


feat = np.maximum(0,to_np(sfs[3].features[0]))
feat.shape


# In[37]:


f2=np.dot(np.rollaxis(feat,0,3), py)
f2-=f2.min()
f2/=f2.max()
f2


# In[38]:


f2F = np.ma.masked_where(f2 <= 0.7, f2)


# In[39]:


filter = scipy.misc.imresize(f2F, dx.shape,mode="L")


# In[44]:


f2Filtered = np.ma.masked_where(filter <= 180, filter)


# In[45]:


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
xrange = (minX, maxX)
yrange = (minX, maxY)
print("x range: ", xrange)
print("y range: ", yrange)


# In[46]:

'''
fig=plt.figure(figsize=(20, 20))
fig.add_subplot(1,2, 1)
plt.imshow(f2Filtered)
fig.add_subplot(1,2,2)
plt.imshow(filter)
plt.show()
'''

# In[47]:

fig=plt.figure(figsize=(20, 20))
fig.add_subplot(1,2, 1)
plt.imshow(dx)
plt.imshow(f2Filtered, alpha=0.7, cmap='hot');
plt.show()





# ## Model
