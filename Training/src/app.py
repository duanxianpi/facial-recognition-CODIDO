import argparse
import zipfile
import os
import glob
import boto3
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

yolov5face = __import__("yolov5-face.main")

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Faces to be train")
parser.add_argument("--output", help="output")
parser.add_argument("--codido", help="running on codido")
############################################
# TODO: add extra args here
############################################

args = parser.parse_args()

input_folder_path = os.path.join(os.sep, 'app/src', 'inputs')
output_folder_path = os.path.join(os.sep, 'app/src', 'outputs')
images_folder_path = os.path.join(str(ROOT)+"/yolov5-face/", 'images')
os.makedirs(input_folder_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)


############################################
# TODO: the input is now accessible via input_file_path
############################################

if args.codido == 'True':
    import boto3
    s3 = boto3.client('s3')

    # downloads codido input file into the folder specified by input_folder_path
    input_file_path = os.path.join(input_folder_path, args.input.split('_SPLIT_')[-1])
    s3.download_file(os.environ['S3_BUCKET'], args.input, input_file_path)
else:
    input_file_path = glob.glob(os.path.join(input_folder_path, '*'))[0]

with zipfile.ZipFile(input_file_path, 'r') as zip_ref:
    zip_ref.extractall(images_folder_path)
os.remove(input_file_path) # remove the zip file

yolov5face.main.main_entry(str(ROOT)+"/yolov5-face/weights/yolov5m-face.pt",str(ROOT)+"/yolov5-face/images",str(ROOT)+"/yolov5-face/temp",str(output_folder_path)+"/Database.npz",str(output_folder_path)+"/SVCmodel.pkl")

if args.codido == 'True':
    # create zip with all the saved outputs
    s3 = boto3.client('s3')

    zip_name = output_folder_path + '.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(output_folder_path):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, output_folder_path))

    # upload
    s3.upload_file(zip_name, os.environ['S3_BUCKET'], args.output)