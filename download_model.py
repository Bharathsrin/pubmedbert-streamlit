# download_model.py

import os
import gdown

def download_model():
    model_path = "pytorch_model.bin"
    if not os.path.exists(model_path):
        print("Downloading model...")
        url = "https://drive.google.com/uc?id=1M-i8hQOt9JMICttv2eFYXFqbaJorcuIJ"
        output = model_path
        gdown.download(url, output, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists. Skipping download.")
