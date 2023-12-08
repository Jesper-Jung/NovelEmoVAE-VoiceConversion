#!/bin/bash

echo welcome to the ESD!

FILEID='1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v'
FILENAME='./data/ESD_Dataset/ESD.zip'

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id="$FILEID > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$FILEID" -o $FILENAME


