# Comparison-RMT-PCA
The code used to train and analyze networks for RMT and PCA in tensorflow

# PCA.ipynb
Given a dynamic weight matrix and a network, the notebook computes the principal components by finding the eigenvalues of the covariance matrix and creates plots in order to analyze the drift mode.

# RMT.ipynb
Given a network with already computed principal components, the notebook computes the singular values, plots a histogram of them and gives a colorplot for the principal componentes to the singular values.
