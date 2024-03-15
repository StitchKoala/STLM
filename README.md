# STLM

The code of paper: A SAM-guided Two-stream Lightweight Model for Anomaly Detection. 

Install Mobile Segment Anything to 'mobile_sam' folder. We employ the image encoder and modify the mask decoder, using two-layer features and removing some unneeded codes.

Visualization results

<p float="center">
  <img src="images/visual.png?raw=true" width="99.1%" />
</p>


## Getting started

### 1) Clone the repository
```
git clone https://github.com/StitchKoala/STLM.git
cd STLM
```

### 2) Download pretrained checkpoints and datasets
```
chmod +x ./preparing.sh
./preparing.sh
```

### 3) Training

```
python trainSTLM.py
```

### 4) Evaluation

```
python evalSTLM.py
```