################################################################################
#
# File
#
#     Code01CNNStyle2DConvolution.py
#
# Purpose
#
#     Converts CNN style 2D convolution to matrix matrix multiplication
#
# Notes
#
#     1. The purpose of this code is clarity (not efficiency) in the
#        transformation of CNN style 2D convolution to matrix matrix
#        multiplication
#
#
#     2. Logical descriptions of where and how to include
#
#        Input feature map zero padding
#        Input feature map up sampling
#        Input filter coefficient up sampling
#        Output feature map down sampling
#
#        are included but not implemented
#
################################################################################


################################################################################
#
# History
#
#     A. Redfern    2018-08-31    Created
#     A. Redfern    2018-09-04    Minor updates for formatting and pre post
#                                 processing
#
################################################################################


################################################################################
#
# Import
#
################################################################################

# numpy
import numpy as np


################################################################################
#
# Constants
#
################################################################################

# data type
DATA_TYPE_SEQUENTIAL = 0
DATA_TYPE_RANDOM     = 1


################################################################################
#
# User Parameters
#
################################################################################

# data type
dataType = DATA_TYPE_SEQUENTIAL

# pre processing

#
# add zero padding parameters
# add filter coefficient up sampling parameters
# add feature map up sampling parameters
# note: various tensor and matrix sizes will change based on pre processing
#

# input
Ni = 2
Lr = 5
Lc = 5

# filter
Fr = 3
Fc = 3

# output
No = 3

# post processing

#
# add feature map down sampling parameters
# note: various tensor and matrix sizes will change based on post processing
#


################################################################################
#
# Derived Parameters
#
################################################################################

# output
Mr = Lr - Fr + 1
Mc = Lc - Fc + 1


################################################################################
#
# Create Data Structures
#
################################################################################

# input and input filtering matrix
X    = np.zeros((Ni, Lr, Lc))
Xmat = np.zeros((Ni*Fr*Fc, Mr*Mc))

# filter (No x Ni x Fr x Fc)
H    = np.zeros((No, Ni, Fr, Fc))
Hmat = np.zeros((No, Ni*Fr*Fc))

# output (No x Mr x Mc)
Y    = np.zeros((No, Mr, Mc))
Ymat = np.zeros((No, Mr*Mc))


################################################################################
#
# Fill In Tensors
#
################################################################################

# sequential
if dataType == DATA_TYPE_SEQUENTIAL:
    
    # input
    val = 0
    for ni in range(Ni):
        for lr in range(Lr):
            for lc in range(Lc):
                X[ni, lr, lc] = val
                val           = val + 1

    # filter
    val = 0
    for no in range(No):
        for ni in range(Ni):
            for fr in range(Fr):
                for fc in range(Fc):
                    H[no, ni, fr, fc] = val
                    val               = val + 1

# random
# elif dataType == DATA_TYPE_RANDOM:

#
# fill in random data
#


################################################################################
#
# Filter:  Tensor Up Sampling
#
################################################################################

#
# insert zeros in rows and cols and update Fr and Fc accordingly
#


################################################################################
#
# Filter:  Create Matrix
#
################################################################################

# filter matrix col index
colIndex = 0

# no is row index
for no in range(No):
    
    # ni, fr and fc combine to determine col index
    for ni in range(Ni):
        for fr in range(Fr):
            for fc in range(Fc):
                Hmat[no, colIndex] = H[no, ni, fr, fc]
                colIndex           = colIndex + 1

    # reset the col
    colIndex = 0

# visualize
print
print("Filter matrix")
print
for rowIndex in range(Hmat[:, 1].size):
    for colIndex in range(Hmat[1, :].size):
        print("{0:8.1f}".format(Hmat[rowIndex, colIndex])),
    print
print


################################################################################
#
# Input:  Tensor Up Sampling
#
################################################################################

#
# insert zeros in rows and cols and update Lr, Lc, Mr and Mc accordingly
#


################################################################################
#
# Input:  Tensor Zero Padding
#
################################################################################

#
# insert zeros around border and update Lr, Lc, Mr and Mc accordingly
#


################################################################################
#
# Input:  Create Matrix
#
################################################################################

# input matrix row col index
rowIndex = 0
colIndex = 0

# mr and mc combine to determine col index
for mr in range(Mr):
    for mc in range(Mc):
        
        # ni, fr and fc combine to determine row index
        for ni in range(Ni):
            for fr in range(Fr):
                for fc in range(Fc):
                    Xmat[rowIndex, colIndex] = X[ni, mr + fr, mc + fc]
                    rowIndex                 = rowIndex + 1
    
        # move to the next col and reset the row
        rowIndex = 0
        colIndex = colIndex + 1

# visualize
print
print("Input feature map filtering matrix")
print
for rowIndex in range(Xmat[:, 1].size):
    for colIndex in range(Xmat[1, :].size):
        print("{0:8.1f}".format(Xmat[rowIndex, colIndex])),
    print
print


################################################################################
#
# Input And Filter:  Remove Rows And Cols From Matrices
#
################################################################################

#
# filter upsampling resuls in many cols of the filter matrix that are all 0
#    remove associated cols from the filter matrix and update sizes accordingly
#    remove associated rows from the input matrix and update sizes accordingly
#    this has the benefit of removing the computations
#

#
# output feature map down sampling results in many cols of the output matrix that are not needed
#    remove associated cols from the input matrix and update sizes accordingly
#    this has the benefit of removing the computations
#


################################################################################
#
# Output:  Create Matrix
#
################################################################################

# inner product based matrix matrix multiplication
for m in range(No):
    for n in range(Mr*Mc):
        for k in range(Ni*Fr*Fc):
            Ymat[m, n] = Ymat[m, n] + Hmat[m, k]*Xmat[k, n]

# visualize
print
print("Output feature map matrix")
print
for rowIndex in range(Ymat[:, 1].size):
    for colIndex in range(Ymat[1, :].size):
        print("{0:8.1f}".format(Ymat[rowIndex, colIndex])),
    print
print


################################################################################
#
# Check
#
################################################################################

# Hrow0 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
# Xcol0 = np.array([0, 1, 2, 5, 6, 7, 10, 11, 12, 25, 26, 27, 30, 31, 32, 35, 36, 37])
# Y00   = np.dot(Hrow0, Xcol0)

# Y00

# Hrow1 = Hrow0 + 18
# Xcol1 = Xcol0 + 1
# Y11   = np.dot(Hrow1, Xcol1)

# Y11

