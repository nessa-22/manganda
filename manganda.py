# cd '.\Desktop\AIM\Term 5\MLOps\Final Project\ManGanda' | streamlit run .\manganda.py

import pandas as pd
import streamlit as st
from io import BytesIO
import pickle
import torch
import torch.nn as nn
import joblib, os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from danbooru_resnet import *
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

# class MangaModel(nn.Module):
#     """Regression Model for Rating Mangas"""
#     def __init__(self, dataset, dataloader, num_epochs=500, save_file='saves/MangaModel.pth'):
#         super(MangaModel, self).__init__()
#         """Initialize with the trained parameters, else retrain from scratch
#         """        
#         # Get the device for loading the model
#         self.device = torch.device('cuda' if torch.cuda.is_available()
#                                    else 'cpu')
#         self.model = resnet18(pretrained=False)
#         self.model.to(self.device)
        
#         # Embed the dataset and dataloader
#         self.dataset = dataset
#         self.dataloader = dataloader
            
#         # Freezing the weights of the pretrained model
#         for param in self.model[0].parameters():
#             param.requires_grad = False
#         self.model[1][8] = nn.Linear(512, 1)
#         self.model[1].append(nn.Threshold(0, 10))
            
#         self.model.to(self.device)
# #         summary(self.model, (3, 224, 224))
        
#         # Set the loss function for Regression
#         self.criterion = nn.MSELoss()

#         # Only the parameters of the regressor are being optimized
#         self.optimizer = optim.Adam(self.model[1].parameters(), lr=0.001)
        
#         # Load the retrained model, else, train
#         try:
#             self.model.load_state_dict(torch.load(save_file))
#         except:
#             self.model = train_model(self,
#                                      self.criterion,
#                                      self.optimizer,
#                                      num_epochs=num_epochs)
        
        
#     def forward(self, X):
#         return self.model(X)

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
        

torch.hub.load('RF5/danbooru-pretrained', 'resnet18', pretrained=False)

if __name__ == "__main__":
    st.image('manganda.png', width=800)
#     model_path = os.path.abspath('model.pth')
    model = torch.load('model.pth')
#     dataset = MangaDataset()
#     dataloader = MangaDataloader(dataset, batch_size=24)
#     model = MangaModel(dataset, dataloader)


    file_upload()
