import argparse
import zipfile
import os
import glob
import boto3
from pathlib import Path
import sys
import time
import psutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import main

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input")
parser.add_argument("--output", help="output")
parser.add_argument("--codido", help="running on codido")
############################################
# TODO: add extra args here
############################################

args = parser.parse_args()

# input_folder_path = os.path.join(os.sep, 'home/duanxianpi/facial-recognition/Using/src', 'inputs')
# output_folder_path = os.path.join(os.sep, 'home/duanxianpi/facial-recognition/Using/src', 'outputs')
input_folder_path = os.path.join(os.sep, 'app/src', 'inputs')
output_folder_path = os.path.join(os.sep, 'app/src', 'outputs')
images_folder_path = os.path.join(str(ROOT)+"/", 'inputs')
os.makedirs(input_folder_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)


############################################
# TODO: the input is now accessible via input_file_path
############################################

print("Step 1")

if args.codido == 'True':
    import boto3
    s3 = boto3.client('s3')

    # downloads codido input file into the folder specified by input_folder_path
    input_file_path = os.path.join(input_folder_path, args.input.split('_SPLIT_')[-1])
    s3.download_file(os.environ['S3_BUCKET'], args.input, input_file_path)
else:
    input_file_path = glob.glob(os.path.join(input_folder_path, '*'))[0]

print("Step 2")

with zipfile.ZipFile(input_file_path, 'r') as zip_ref:
    zip_ref.extractall(images_folder_path)
os.remove(input_file_path) # remove the zip file

print("Step 3")

input_files_path = glob.glob(os.path.join(input_folder_path, '*'))

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

source = ""

for file in input_files_path:
    if (Path(file).suffix[1:] in (img_formats + vid_formats)):
        source = file

npz_file = glob.glob(os.path.join(input_folder_path, '*.npz'))[0]
pkl_file = glob.glob(os.path.join(input_folder_path, '*.pkl'))[0]

print("Step 4")

if (source == "" or npz_file == "" or pkl_file == ""):
    print("Invalid Input")
    sys.exit()

print("Step 5")

pid = os.getpid()
p = psutil.Process(pid)
info_start = p.memory_full_info().uss/1024/1024
start_time = time.time()
main.main_entry(source,pkl_file,npz_file)
over_time = time.time()
info_end=p.memory_full_info().uss/1024/1024
total_time = over_time - start_time

with open(os.path.join(str(ROOT)+"/outputs", 'log'), 'w') as f:
    f.write("Took"+str(info_end-info_start)+"MB, " + str(total_time) + "s")

print("Step 6")

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