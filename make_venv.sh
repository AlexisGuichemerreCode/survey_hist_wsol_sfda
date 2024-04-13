#!/usr/bin/env bash
# Run script: ./make_venv.sh NAME_OF_YOUR_VENV
#env=da
env=$1

echo "Deleting existing env $env"
conda remove --name $env --all
rm -rf ~/anaconda3/envs/$env

echo "Create env $env"
conda create -n $env python=3.10

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $env

python --version

cdir=$(pwd)


echo "Installing..."
# pip install  --upgrade pip

pip install -r reqs.txt

pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install pretrainedmodels zipp timm==0.6.13 kornia
pip install efficientnet-pytorch==0.7.1

pip install seaborn

pip install gdown
pip install pykeops==2.1.2

cd $cdir

cd $cdir/dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
pip install .

cd $cdir
cd dlib/crf/crfwrapper/colorbilateralfilter
swig -python -c++ colorbilateralfilter.i
pip install .

conda deactivate

echo "Done creating and installing virt.env: $env."