#!/usr/bin/env python2
import math, sys, os, copy, shutil, struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
os.environ['GLOG_minloglevel']='3'
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
from Tools import *

def GramSchmidt(Vecs, Orth=False):
    Basis    = copy.deepcopy(Vecs)
    Dim      = Basis.shape[0]
    TransMat = np.zeros((Dim,Dim), dtype='float')
    for j in range(1,Dim):
        for k in range(0,j):
            Basis[j] = Basis[j]-np.dot(Basis[k],Basis[j])/np.dot(Basis[k],Basis[k])*Basis[k]
    if Orth:
        for i in range(Dim):
            Basis[i] = Basis[i]/np.linalg.norm(Basis[i])
    for i in range(Dim):
        for j in range(Dim):
            TransMat[i][j] = np.dot(Vecs[i], Basis[j])/np.dot(Basis[j], Basis[j])
    return Basis, TransMat

def CalPrunedBasisGramSchmidt(Vecs, Norm=False, PruneTh=0.01, OrthTh=0.3):
    #print("[I]***CalPrunedBasisGramSchmidt***")
    Basis    = copy.deepcopy(Vecs)
    VecsNum  = Vecs.shape[0]
    MaxNorm  = 0
    Ctr      = 1

    #Cal. norms and sort Vecs by norm
    VecNorms = np.zeros(VecsNum, dtype='float')
    for i in range(VecsNum):
        VecNorms[i] = -np.linalg.norm(Vecs[i]) #!!!sort the index from small to large
    SortedVecNormsIndex = np.argsort(VecNorms)

    #Cal. basis by GramSchmidt method
    Drop       = False
    Basis[0]   = Vecs[int(SortedVecNormsIndex[0])]
    MaxVecNorm = np.linalg.norm(Basis[0])
    for j in range(1, VecsNum):
        Index   = int(SortedVecNormsIndex[j])
        VecNorm = np.linalg.norm(Vecs[Index])
        Basis_  = Vecs[Index]
        for k in range(Ctr):
            Basis_ = Basis_-np.dot(Basis[k],Basis_)/np.dot(Basis[k],Basis[k])*Basis[k]
        #Drop the basis which norm small than threshold
        BasisNorm = np.linalg.norm(Basis_)
        #print("[I]: MaxNorm={},BasisNorm={}".format(MaxNorm,BasisNorm))
        if VecNorm >= PruneTh*MaxVecNorm and BasisNorm >= OrthTh*np.linalg.norm(Vecs[Index]):
            Basis[Ctr] = Basis_
            Ctr       += 1
        else:
            #print("[I]: basis from Vecs["+str(j)+"] droped, BasisNorm="+str(BasisNorm)+" < MaxNorm*Threshold="+str(Threshold*MaxNorm) )
            Drop = True
    BasisDim = Ctr
    Basis    = Basis[0:Ctr]

    #Orthogonalization
    if Norm:
        for i in range(BasisDim):
            Basis[i] = Basis[i]/np.linalg.norm(Basis[i])

    #Cal. coordinates
    TransMat = np.zeros((VecsNum, BasisDim), dtype='float')
    for i in range(VecsNum):
        for j in range(BasisDim):
            TransMat[i][j] = np.dot(Vecs[i], Basis[j])/np.dot(Basis[j], Basis[j])
    #if Drop:
    #    print("[I]: The difference Vecs-TransMat(/dot)Basis is:\n{}".format(Vecs - np.dot(TransMat, Basis)) )
    return Basis, TransMat


def OrthogonalizationV2(ExpandCoeffs, Weights, Norm=False, PruneTh=1.e-2, OrthTh=0.3):
    C      = len(ExpandCoeffs)
    DimVec = Weights.shape[1]*Weights.shape[2]
    OrthonormalBasisArr = []
    TransMatArr         = []
    Ctr    = 0
    for i in range(C):
        Vecs = np.zeros((ExpandCoeffs[i], DimVec), dtype='float')
        for iEc in range(ExpandCoeffs[i]):
            Vecs[iEc] = copy.deepcopy(Weights[Ctr]).reshape(DimVec)
            Ctr      += 1
        OrthonormalBasis, TransMat = CalPrunedBasisGramSchmidt(Vecs, Norm=Norm, PruneTh=PruneTh, OrthTh=OrthTh)
        OrthonormalBasisArr.append(OrthonormalBasis)
        TransMatArr.append(TransMat)
    return OrthonormalBasisArr, TransMatArr

def GetSparseConvLayerWeights(ModelPath, WeightsPath, LayerNameEnds="SparseConv"):
    # Load source model
    Model = caffe_pb2.NetParameter()
    with open(ModelPath) as File:
        s = File.read()
        text_format.Merge(s, Model)
    File.close()

    net = caffe.Net(ModelPath, WeightsPath, caffe.TEST)
    AllSparseConvLayerWeights=[]
    for param_name in net.params.keys():
        Weights      = []
        if param_name.endswith(LayerNameEnds):
						#LayerName
            Weights.append(param_name)

            ExpandCoeffs = []
            #Search sparse convolution expand paramter
            for Layer in Model.layer:
                if Layer.name==param_name:
                    ChannelExpandFilePath  = Layer.sparse_convolution_param.channel_expand_file
                    if ChannelExpandFilePath != "":
                        ChannelExpandFile = open(ChannelExpandFilePath, 'rb')
                        InputChannel      = int(struct.unpack("f", ChannelExpandFile.read(4))[0])
                        for i in range(InputChannel):
                            Data = int(struct.unpack("f", ChannelExpandFile.read(4))[0])
                            ExpandCoeffs.append(Data)
                        ChannelExpandFile.close()
                    else:
                        ChannelExpandMultiplier = int(Layer.sparse_convolution_param.channel_expand_multiplier)
                        InputChannel            = int(net.params[param_name][0].data.shape[0] / int(ChannelExpandMultiplier))
                        for i in range(InputChannel):
                            ExpandCoeffs.append(ChannelExpandMultiplier)
                    Weights.append(ExpandCoeffs)
	
            #Sparse convolution kernels
            for Slice in net.params[param_name]:
                weight = Slice.data
                Weights.append(weight)
						
            AllSparseConvLayerWeights.append(Weights)
    return AllSparseConvLayerWeights

