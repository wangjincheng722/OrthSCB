#!/usr/bin/env python2
from __future__ import print_function
import sys, os, math, copy
caffe_path="/home/jcwang/work/0.SparseConv/0.SparseCaffeV9.1"
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np

def make_solver(SolverPath, ModelPath, MaxStep, BaseLR, GlobalWeightDecay, TrainWeightPath):
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE
    s.net = ModelPath
    s.test_interval = max(MaxStep/500, 5000)
    s.test_iter.append(10000)
    s.max_iter = MaxStep
    s.lr_policy = 'multistep'
    s.stepvalue.append(int(MaxStep*0.5))
    s.stepvalue.append(int(MaxStep*0.75))
    s.stepvalue.append(int(MaxStep*0.875))
    s.display = 20
    s.snapshot = max(MaxStep/100, 10000)
    s.snapshot_prefix = TrainWeightPath+'/FT'
    s.base_lr = BaseLR
    s.momentum = 0.9
    s.weight_decay = GlobalWeightDecay
    s.gamma = 0.1
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    s.type = 'SGD'
    solver_path = SolverPath
    with open(solver_path, 'w') as solver_file:
        solver_file.write(str(s))
    return

def make_run_script(TaskPath, ModelPath, ModelParamPath, GPUID):
    log_file = os.path.join(TaskPath, 'solver.log')
    solver_path = os.path.join(TaskPath, 'solver.pt')
    run_s = 'netname='+ModelPath+';\nrm ' + log_file + ';\nnohup ' + str(caffe_path) + '/build/tools/caffe train --solver=' + solver_path + ' --weights='+ModelParamPath +' --gpu '+str(GPUID)+' > '+str(log_file)+'&\n'
    script_path = os.path.join(TaskPath, 'run.sh')
    with open(script_path, 'w') as script_file:
        script_file.write(run_s)
    command = 'chmod +x ' + script_path +';'
    os.system(command)
    return
