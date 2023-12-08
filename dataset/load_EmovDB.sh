#!/bin/bash

echo welcome to the Emov-DB!

LIST_PTH="https://www.openslr.org/resources/115"

LIST_PTH_NAME="
/bea_Amused.tar.gz
/bea_Angry.tar.gz
/bea_Disgusted.tar.gz
/bea_Neutral.tar.gz
/bea_Sleepy.tar.gz
/jenie_Amused.tar.gz
/jenie_Angry.tar.gz
/jenie_Disgusted.tar.gz
/jenie_Neutral.tar.gz
/jenie_Sleepy.tar.gz
/sam_Amused.tar.gz
/sam_Angry.tar.gz
/sam_Disgusted.tar.gz
/sam_Neutral.tar.gz
/sam_Sleepy.tar.gz
/josh_Amused.tar.gz
/josh_Neutral.tar.gz
/josh_Sleepy.tar.gz
"

_list_spk_name="
bea
bea
bea
bea
bea
jenie
jenie
jenie
jenie
jenie
sam
sam
sam
sam
sam
josh
josh
josh
"
LIST_SPK_NAME=($_list_spk_name)
NAMES=("bea" "jenie" "sam" "josh")


PTH_ROOT="/home/jung3388/HYUN_lab"
PTH_DIR="/data/EmovDB_Dataset"

for name in "${NAMES[@]}"; do
	mkdir $PTH_ROOT$PTH_DIR"/"$name
done

i=0
for PTH_NAME in $LIST_PTH_NAME; do
    echo "PTH" $i ":" $LIST_PTH$PTH_NAME
    wget $LIST_PTH$PTH_NAME -P $PTH_ROOT$PTH_DIR
    tar -xvf $PTH_ROOT$PTH_DIR$PTH_NAME -C $PTH_ROOT$PTH_DIR"/"${LIST_SPK_NAME[${i}]}

    i=`expr $i + 1`
done