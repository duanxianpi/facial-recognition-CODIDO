# Download zip dataset from Google Drive
yolov5='yolov5m-face.pt'
yolov5_fileid='1Sx-KEGXSxvPMS35JhzQKeRBiqC98VDDI'

facenet='facenet.zip'
facenet_fileid='1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-'

wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${facenet_fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${facenet_fileid}" -O ${facenet} && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${yolov5_fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${yolov5_fileid}" -O ${yolov5} && rm -rf /tmp/cookies.txt

# Unzip
unzip -q ${facenet}
rm ${facenet}