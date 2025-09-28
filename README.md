#  Learning Efficient Meshflow and Optical Flow From Event Cameras.
<h4 align="center">Xionglong Luo<sup>1</sup>, Ao Luo<sup>2</sup>, Kunming Luo<sup>3</sup>, Zhengning Wang<sup>1</sup>, Ping Tan<sup>3</sup>, Bing Zeng<sup>1</sup>, Shuaicheng Liu<sup>1</sup></center>
<h4 align="center">1.University of Electronic Science and Technology of China, 
<h4 align="center">2.Southwest Jiaotong University,
<h4 align="center">3.The Hong Kong University of Science and Technology </center></center>

## Environments
You will have to choose cudatoolkit version to match your compute environment. The code is tested on Python 3.7 and PyTorch 1.10.1+cu113 but other versions might also work. 
```bash
conda create -n EEMFlow python=3.7
conda activate EEMFlow
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
## Dataset
### MVSEC
You need download the HDF5 files version of [MVSEC](https://daniilidis-group.github.io/mvsec/download/) datasets. We provide the code to encode the events and flow label of MVSEC dataset.
```python
# Encoding Events and flow label in dt1 setting
python loader/MVSEC_encoder.py --only_event -dt=1
# Encoding Events and flow label in dt4 setting
python loader/MVSEC_encoder.py --only_event -dt=4
# Encoding only Events
python loader/MVSEC_encoder.py --only_event
```

### HREM
This work proposed  a large-scale High-Resolution Event Meshflow (HREM+) dataset, you can download it from [https://pan.baidu.com/s/1iSgGCjDask-M_QqPRtaLhA?pwd=z52j](https://pan.baidu.com/s/1v41gwHEEiSOKLkziTEYANQ?pwd=kcps ) .

## Evaluate
### Pretrained Weights
Pretrained weights can be downloaded from 
[Google Drive](https://drive.google.com/drive/folders/15uwhrmUzg3kK3UB6z0Qnht-sGs7Nq23o?usp=sharing).
Please put them into the `checkpoint` folder.

### Test on HREM+
```python
python test_EEMFlowPlus_HREM.py -dt dt1
python test_EEMFlowPlus_HREM.py -dt dt4
```

## Acknowledgments

Thanks the assiciate editor and the reviewers for their comments, which is very helpful to improve our paper. 

Thanks for the following helpful open source projects:
[ERAFT](https://github.com/uzh-rpg/E-RAFT),
[TMA](https://github.com/ispc-lab/TMA),
[BFlow](https://github.com/uzh-rpg/bflow).
