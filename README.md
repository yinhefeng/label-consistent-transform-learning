# label-consistent-transform-learning
LCTL solves the following optimization problem,<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{\mathbf{T},\mathbf{Z},\mathbf{A}}{\textrm{min}}\left&space;\|&space;\mathbf{T}\mathbf{X}-\mathbf{Z}&space;\right&space;\|_F^2&plus;\mu(\left&space;\|&space;\mathbf{T}&space;\right&space;\|_F^2-\textrm{log}&space;\&space;\textrm{det}&space;\&space;\mathbf{T})&plus;\alpha\left&space;\|&space;\mathbf{Q}-\mathbf{A}\mathbf{Z}&space;\right&space;\|_F^2&plus;\lambda\left&space;\|&space;\mathbf{Z}&space;\right&space;\|_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{\mathbf{T},\mathbf{Z},\mathbf{A}}{\textrm{min}}\left&space;\|&space;\mathbf{T}\mathbf{X}-\mathbf{Z}&space;\right&space;\|_F^2&plus;\mu(\left&space;\|&space;\mathbf{T}&space;\right&space;\|_F^2-\textrm{log}&space;\&space;\textrm{det}&space;\&space;\mathbf{T})&plus;\alpha\left&space;\|&space;\mathbf{Q}-\mathbf{A}\mathbf{Z}&space;\right&space;\|_F^2&plus;\lambda\left&space;\|&space;\mathbf{Z}&space;\right&space;\|_1" title="\underset{\mathbf{T},\mathbf{Z},\mathbf{A}}{\textrm{min}}\left \| \mathbf{T}\mathbf{X}-\mathbf{Z} \right \|_F^2+\mu(\left \| \mathbf{T} \right \|_F^2-\textrm{log} \ \textrm{det} \ \mathbf{T})+\alpha\left \| \mathbf{Q}-\mathbf{A}\mathbf{Z} \right \|_F^2+\lambda\left \| \mathbf{Z} \right \|_1" /></a>

where X is the training data matrix, Q is the ideal representation matrix, A is a matrix that transform the learned coefficients to Q, T is the learned transform matrix and Z is the learned coefficient matrix.

The difference between [1] and our work is that, [1] actually introduces the classification error into the original transform learning.

The feature of Scene 15 dataset can be downloaded from https://pan.baidu.com/s/13fLCEhZONv_CtroZnacT1w or http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html

[1] Maggu J, Aggarwal H K, Majumdar A. Label-Consistent Transform Learning for Hyperspectral Image Classification[J]. IEEE Geoscience and Remote Sensing Letters, 2019.

[2] He-Feng Yin, Xiao-Jun Wu. Label consistent transform learning for pattern classification. Journal of Algorithms & Computational Technology, 2019, 13: 1748302619881395.
