# The main entery for the software

from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import train
import detect_face
import facenet
import torch

def main_entry(yolov5_model_path, picture_path, cropped_picture_path, database_path, SVCpath):
    # Set the paths
    model_path = str(ROOT)+"/weights/20180402-114759/20180402-114759.pb"  #facenet weight path

    # picture_path = "./images" #face to crop
    # cropped_picture_path = "./temp" #face to train
    # database_path = "./npz/Database.npz"  #pack to npz

    # Step 1. detect the face from the images and crop to 640*640

    ## Get all image path
    all_image_paths = []
    path_exp = os.path.expanduser(picture_path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = facenet.get_image_paths(facedir)
        all_image_paths.append([image_paths,classes[i]])



    ## Detect the face and crop the picture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = detect_face.load_model(yolov5_model_path, device)

    with open(os.path.join(str(ROOT)+"/../outputs", 'log'), 'w') as f:
        f.write(str(device))

    for image_paths in all_image_paths:
        for image_path in image_paths[0]:
            detect_face.detect(model, image_path,device,str(ROOT)+"/temp",image_paths[1],True,True,False)

    # Step 2. Extract FaceNet to database
    train.face2database(cropped_picture_path,model_path,database_path) #step1

    # Step 3. Train the SVM to classify the face from database
    # SVCpath = "./pkl/SVCmodel.pkl"
    train.ClassifyTrainSVC(database_path,SVCpath)