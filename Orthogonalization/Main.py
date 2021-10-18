#!/usr/bin/env python2
from __future__ import print_function
import os, sys, math, time
from GenOrthSparseConvNet import *

#Train Parameters
ModelPath  = sys.argv[1]
DeployPath = sys.argv[2]
ParaPath   = sys.argv[3]
if len(sys.argv)==6:
    PruneTh    = float(sys.argv[4])
    OrthTh     = float(sys.argv[5])
    TaskPath   = "PruneTh{}OrthTh{}".format(str(sys.argv[4]), str(sys.argv[5]))
if len(sys.argv)==5:
    PruneTh    = 0
    OrthTh     = float(sys.argv[4])
    TaskPath   = "OrthTh{}".format(str(sys.argv[4]))
LR         = 1.e-3
FTSteps    = 4000000
GPUID      = 0
GlobalWeightDecay = 1.e-5
CreateOrthTrainJob(ModelPath, DeployPath, ParaPath, TaskPath, LR, FTSteps, GlobalWeightDecay, PruneTh=PruneTh, OrthTh=OrthTh, GPUID=GPUID)
