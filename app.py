import streamlit as st 
from fastai.vision.all import *
from fastai.vision.widgets import *
#from fastai.vision import image
from tensorflow.keras.preprocessing import image
import pickle
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
from keras.preprocessing.image import load_img


st.header("Product Label Detection")
st.text("--"*20)
labels = pd.read_csv('images.csv')
labels.loc[labels['label']=='Not sure','label'] = 'Not_sure'
labels['image'] = labels['image'] + '.jpg'
labels['label_cat'] = labels['label'].astype(str)
label_df = labels[['image', 'label_cat']]

def get_x(r): return 'images_compressed/'+r['image'] # create path to open images in the original folder
def get_y(r): return r['label_cat'].split(' ') # split the labels using space as a delimitter

# Create DataBlock
dblock = DataBlock(blocks = (ImageBlock, MultiCategoryBlock),
                  get_x = get_x, get_y = get_y,
                  item_tfms = RandomResizedCrop(128, min_scale=0.35))  # ensure every item is of the same size
dls = dblock.dataloaders(label_df)

m1 = load_learner("export.pkl",'rb')

img_ = st.file_uploader("Upload an Image",type=["png","jpeg","jpg"])
if img_ is None:
    st.text("Please upload an Image")
else:
    im_g = Image.open(img_)
    #img = cv2.imread(img)
    img = np.array(im_g)
    dim =(128,128)
    img = cv2.resize(img, dim)
    pred = m1.predict(img)
    st.write("Item is : ",pred[0])
    st.image(img_)
