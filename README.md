# MPS-FFA: A multiplane and multiscale feature fusion attention network for Alzheimer’s disease prediction with structural MRI

 - It is my pleasure that if this project may give you a little inspiration. Hope you can follow it and do better works. :blush:
 - The paper can be found at [here](https://www.sciencedirect.com/science/article/abs/pii/S001048252300255X)

# The pipeline of the MPS-FFA
![Pipeline of the MPSFFA](overall_framework.jpg)

## Abstract 
Structural magnetic resonance imaging (sMRI) is a popular technique that is widely applied in Alzheimer’s disease (AD) diagnosis. However, only a few structural atrophy areas in sMRI scans are highly associated with AD. The degree of atrophy in patients’ brain tissues and the distribution of lesion areas differ among patients. Therefore, a key challenge in sMRI-based AD diagnosis is identifying discriminating atrophy features. Hence, we propose a multiplane and multiscale feature-level fusion attention (MPS-FFA) model. The model has three components, (1) A feature encoder uses a multiscale feature extractor with hybrid attention layers to simultaneously capture and fuse multiple pathological features in the sagittal, coronal, and axial planes. (2) A global attention classifier combines clinical scores and two global attention layers to evaluate the feature impact scores and balance the relative contributions of different feature blocks. (3) A feature similarity discriminator minimizes the feature similarities among heterogeneous labels to enhance the ability of the network to discriminate atrophy features. The MPS-FFA model provides improved interpretability for identifying discriminating features using feature visualization. The experimental results on the baseline sMRI scans from two databases confirm the effectiveness (e.g., accuracy and generalizability) of our method in locating pathological locations. The source code is available at https://github.com/LiuFei-AHU/MPSFFA.

## Keywords
Alzheimer’s disease, sMRI, Multiplane and multiscale, Attention fusion, Convolutional neural network


## How to run the code
Here below shows how to run this demo code.:stuck_out_tongue_winking_eye:_
1. create conda environment <br>
conda create -n mpsffa
2. install Pytorch cuda <br>
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
3. install required lib<br>
pip install -r requirements.txt

Note: Maybe you would better customize the dataset or dataloader implementation according to your private data structure.

## How to train
```python main.py```

## How to test
```python test.py```

## Data and Dataset details
The dataset description and the pre-process operation will be provided soon.
TODO


## Citation

If you want to use the code in your research or wish to refer to the results published in the paper, please use the following BibTeX entry. Thank you very much :kissing:.

```BibTeX
@article{LIU2023106790,
  title = {MPS-FFA: A multiplane and multiscale feature fusion attention network for Alzheimer’s disease prediction with structural MRI},
  journal = {Computers in Biology and Medicine},
  volume = {157},
  pages = {106790},
  year = {2023},
  issn = {0010-4825},
  doi = {https://doi.org/10.1016/j.compbiomed.2023.106790},
  url = {https://www.sciencedirect.com/science/article/pii/S001048252300255X},
  author = {Fei Liu and Huabin Wang and Shiuan-Ni Liang and Zhe Jin and Shicheng Wei and Xuejun Li},
```
F. Liu, H.B. Wang, S.N. Liang, Z. Jin, S.C. Wei, X.J. Li. MPS-FFA: A multiplane and multiscale feature fusion attention network for Alzheimer's disease prediction with structural MRI[J]. Computers in Biology and Medicine, 2023, 157:106790.

## Issue
If you have any questions or problems when using the code in this project, please feel free to send an email to <liu.jason0728@gmail.com> .

