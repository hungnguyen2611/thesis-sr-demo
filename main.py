import gc

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from streamlit_image_comparison import image_comparison
from torchvision.transforms import Compose, ToTensor

from model import decoder, encoder

WEIGHT_PATH = './weights/epoch_4520_weight.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SuperResolution(nn.Module):
    def __init__(self) -> None:
        super(SuperResolution, self).__init__()
        self.model_Enc = encoder.Encoder_RRDB(num_feat=64)
        self.model_Dec_SR = decoder.Decoder_SR_RRDB(num_in_ch=64)

    def forward(self, img):
        feat = self.model_Enc(img)
        output = self.model_Dec_SR(feat)
        return output


class Model(object):
    def __init__(self) -> None:
        self.model_sr = self.load_model(WEIGHT_PATH)
        self.preprocessor = Compose([ToTensor()])

    @staticmethod
    def load_model(weight_path=WEIGHT_PATH):
        weight = torch.load(weight_path, map_location=DEVICE)
        model_sr = SuperResolution().to(DEVICE).eval()
        model_sr.model_Enc.load_state_dict(weight['model_Enc'])
        model_sr.model_Dec_SR.load_state_dict(weight['model_Dec_SR'])
        print("[LOADING] Loading done!")
        del weight
        gc.collect()
        return model_sr

    @staticmethod
    def post_process(out):
        min_max = (0, 1)
        out = out.detach()[0].float().cpu()

        out = out.squeeze().float().cpu().clamp_(*min_max)
        out = (out - min_max[0]) / (min_max[1] - min_max[0])
        out = out.numpy()
        out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))

        out = (out * 255.0).round()
        out = out.astype(np.uint8)
        return out

    def predict(self, img):
        with torch.no_grad():
            img = Image.open(img).convert('RGB')
            # check image shape
            if img.size[0] > 500 and img.size[1] > 500:
                raise ValueError("Image size must be smaller than 300x300")
            img = self.preprocessor(img)
            img = img.unsqueeze(0)
            img = img.to(DEVICE)

            feat = self.model_sr(img)
            output_img = self.post_process(feat)
            return output_img


@st.cache_data
def load_model():
    return Model()


if __name__ == '__main__':
    # create a streamlit app for demo
    st.set_page_config(
        page_title="🔥🔥🔥Super resolution demo 🔥🔥🔥",
        page_icon="🔥",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.markdown(
        """
        <h2 style='text-align: center'>
        🔥🔥🔥 Super Resolution Demo 🔥🔥🔥
        </h2>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style='text-align: center'>
        <a href='https://github.com/hungnguyen2611/super-resolution' target='_blank'>Github</a>
        <br />
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.write("##")
    model = load_model()
    with st.form(key="DEMO"):
        st.markdown(
            """
            <p style='text-align: center'>
            <b>Upload your image</b>
            </p>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])
        submit_button = st.form_submit_button(label="Submit")
        out = None
        if submit_button:
            if uploaded_file is not None:
                st.write("")
                st.write("Processing...")
                try:
                    out = model.predict(uploaded_file)
                except ValueError as e:
                    st.write(e)
                    st.stop()
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                static_component = image_comparison(
                    img1=Image.open(uploaded_file),
                    img2=Image.fromarray(out),
                    label1="Low resolution",
                    label2="Super resolution"
                )
            else:
                st.write("Please upload an image file")
