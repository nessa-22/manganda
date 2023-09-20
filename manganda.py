# cd '.\Desktop\AIM\Term 5\MLOps\Final Project\ManGanda' | streamlit run .\manganda.py

import pandas as pd
import streamlit as st
from io import BytesIO
import pickle
import torch
import joblib, os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# from dataset import MangaDataset
# from dataloader import MangaDataloader
from manganda_util import MangaModel, train_model


STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""


def load_pkl(name, prompt=False):
    """Load an object from a pickle file.
    """
    folder = 'pickles'
    ext = '.pkl'
    if not os.path.exists(folder):
        raise ValueError("'pickles' folder does not exist.")
    
    if name[-4:] == ext:
        fp = os.path.join(folder, name)
    else:
        fp = os.path.join(folder, name+ext)
    pkl = joblib.load(fp)
    
    if prompt:
        print('Pickle file loaded.')
    
    return pkl

def plot_predictions(x, model):
    """Plot sample images and print prediction"""
    means = load_pkl('means')
    stds = load_pkl('stds')

    mean_thresh = np.quantile(means, 0.01)
    std_thresh = np.quantile(stds, 0.01)

    if (x.mean().item() >= mean_thresh and 
        x.std().item() >= std_thresh):
        pass
    else:
        raise Error

    # Group the list of tensors into a batched tensor
    x = torch.stack([torch.tensor(x)])

    x = x.to(model.device)
    out = model(x)

    return out


def file_upload():
    file_types = ["png", "jpg"]

#     st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload file", type=file_types)
    show_file = st.empty()

    if not file:
        show_file.info("Upload manga image: " + ", ".join(file_types))
        return

    content = file.getvalue()
    show_file.image(file)
    
    if st.button("What's the rating?"):
        try:
            image = transforms.ToTensor()(Image.open(file))
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            dim = min(torch.tensor(image.shape[1:])).item()
            transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.CenterCrop(dim),
                transforms.Resize(224, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
            image = transform(image)

            result=plot_predictions(image, model)
            st.subheader(f'Predicted Manga Rating: {result.item():0.2f}')
        except:
            st.subheader('Error: Uploaded Manga Panel is either too dark or primarily white.')

    file.close()
        


if __name__ == "__main__":
    st.image('manganda.png', width=1500)
    model = torch.load('model.pth')

    file_upload()
