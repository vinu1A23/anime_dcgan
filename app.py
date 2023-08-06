import os
from os import listdir
from os.path import join
import random
import urllib
import torch
import streamlit as st
from PIL import Image
import model
from torchvision.utils import save_image
import torchvision
import numpy as np
# Streamlit encourages well-structured code, like starting execution in a main() function.
# from test_video import test_single_video


def get_display_name(image_path):
    image_name = os.path.split(image_path)[1]
    psnr = "psnr"
    ssim = "ssim"
    psnr_pos = image_name.find(psnr)
    ssim_pos = image_name.find(ssim)
    return image_name[:psnr_pos-1] + " with PSNR=" + image_name[psnr_pos+5:ssim_pos-1] + \
        " and SSIM=" + image_name[ssim_pos+5:-4]
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

def main():
    # Render the readme as markdown using st.markdown.
    #readme_text = st.markdown(get_file_content_as_string("instructions.md"))
    ngpu=0
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    st.subheader("Generate Image")
    if st.button('Generate'):
       model_G=model.Generator(0)
       model_G.load_state_dict(torch.load(os.path.join(os.getcwd(),'G.ckpt'),map_location=device))
       model_G.eval()
       fixed_noise = torch.randn(64, 100, 1, 1, device=device)
       with torch.no_grad():
           fake = model_G(fixed_noise)
       save_image(denorm(fake[0]),os.path.join(os.getcwd(),'fake2.jpg'))
       st.image(np.array(torchvision.io.read_image(os.path.join(os.getcwd(),'fake2.jpg'))[0]))




   

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/ENGI9805-COMPUTER-VISION/Term-Project/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


if __name__ == "__main__":
    main()
