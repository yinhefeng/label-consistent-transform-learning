# label-consistent-transform-learning
LCTL solves the following optimization problem,
min_{T,Z,A}||TX - Z||_F^2 - mu*logdet(T) + eps*mu||T||_F^2+ alpha||Q-AZ||_F^2+lambda||Z||_1

where X is the training data matrix, Q is the ideal representation matrix, A is a matrix that transform the learned coefficients to Q, T is the learned transform matrix and Z is the learned coefficient matrix.

The difference between [1] and our work is that, [1] actually introduces the classification error into the original transform learning.

The feature of Scene 15 dataset can be downloaded from https://pan.baidu.com/s/13fLCEhZONv_CtroZnacT1w or http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html

[1] Maggu J, Aggarwal H K, Majumdar A. Label-Consistent Transform Learning for Hyperspectral Image Classification[J]. IEEE Geoscience and Remote Sensing Letters, 2019.

<a href="https://www.codecogs.com/eqnedit.php?latex=ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" title="ax^{2} + by^{2} + c = 0" /></a>
