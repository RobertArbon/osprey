#!/bin/bash
MINICONDA=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_REPO=https://repo.anaconda.com/miniconda
MINICONDA_MD5=$(curl -s $MINICONDA_REPO | grep -A3 $MINICONDA | sed -n '4p' | sed -n 's/ *<td>\(.*\)<\/td> */\1/p')

wget $MINICONDA_REPO/$MINICONDA
if [[ $MINICONDA_MD5 != $(md5sum $MINICONDA | cut -d ' ' -f 1) ]]; then
    echo "Miniconda MD5 mismatch"
    exit 1
fi
bash $MINICONDA -b
rm -f $MINICONDA

export PATH=$HOME/miniconda3/bin:$PATH

conda install -yq python=$CONDA_PY
conda update -yq conda
conda install -yq conda-build jinja2 conda-verify
