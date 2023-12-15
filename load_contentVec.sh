#!/bin/bash

echo Load contextVec pretrained-model.

FILEID='1xpOHdqMZ6WIEW3RaJv_vr7oSwI-eEWZS'
FILENAME='checkpoint_best_legacy_500.pt'

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id="$FILEID > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$FILEID" -o $FILENAME

