

## Before You Start
- make sure you have the latest nvidia drivers https://developer.nvidia.com/cuda-downloads
- install anaconda for managing python environments and packages https://www.anaconda.com/
- create a huggingface token which you will need for auto model download: https://huggingface.co/settings/tokens
- install ffmpeg https://ffmpeg.org/download.html
- install git for your system. you can install git with anaconda:
```
conda install -c anaconda git -y

```

## Getting Started
1. open anaconda powershell (on Windows) or terminal (Linux) and navigate to install location
2. clone the github repository:
```
git clone https://github.com/deforum-art/deforum-stable-diffusion.git
cd deforum-stable-diffusion

```
3. create anaconda environment:
```
conda create -n dsd python=3.10 -y
conda activate dsd
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

```
4. install required packages:
```
python -m pip install -r requirements.txt

```

