from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import face_recognition
import torch

def main_entry(pictue_path,svc_path,npz_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = face_recognition.load_model("weights/yolov5m-face.pt", device)
    face_recognition.detect(model, pictue_path ,device,"outputs","output",True,True,False,svc_path,"weights/20180402-114759/20180402-114759.pb",npz_path)