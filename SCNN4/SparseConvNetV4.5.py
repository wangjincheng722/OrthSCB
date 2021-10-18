#!/usr/bin/env python2
from __future__ import print_function
import os, math
import caffe
import datetime as DT
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format

#net paramter
caffe_path="/home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/5.SparseConvCaffeV9.1"
train_lst="/home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/4.TrainRuns/train.txt"
test_lst="/home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/4.TrainRuns/test.txt"
train_batch=32
test_batch=1
num_channels=3
image_size=32
imgcrop_size=32
ClassNum=10
argument0='CIFAR10'
time_string=DT.datetime.now().strftime('%Y%m%d%H%M%S')

# Solver Parameter
max_step=100000
snapshot=max(max_step/50,50000)
base_lr=0.0001
l1_lambda=0.
gpu_id='0'
global_weight_decay=1.e-5
lr_policy='fixed'
step_ratio=[0.6,0.8,0.9]
def ConvBNScaleReLU(net, from_layer, name, num_output, kernel_size=3, stride=1, pad=1,l1_lambda=l1_lambda, dropout_ratio=0.5):
    exec('net.'+name+'_ConvBNScaleReLU_Conv = L.Convolution(from_layer, name=name+\'_ConvBNScaleReLU_Conv\', kernel_size=kernel_size, stride=stride, num_output=num_output, pad=pad, bias_term=False, weight_filler=dict(type=\'xavier\'), bias_filler=dict(type=\'constant\'))')
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
        'moving_average_fraction': 0.999,
        }
    exec('net.'+name+'_ConvBNScaleReLU_BN = L.BatchNorm(net.'+name+'_ConvBNScaleReLU_Conv, name=name+\'_ConvBNScaleReLU_BN\',in_place=True, **bn_kwargs)')
    exec('net.'+name+'_ConvBNScaleReLU_Scale = L.Scale(net.'+name+'_ConvBNScaleReLU_BN, name=name+\'_ConvBNScaleReLU_Scale\', bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0), l1_lambda=l1_lambda)') #modified scale_layer

    exec('net.'+name+'_ConvBNScaleReLU_ReLU = L.ReLU(net.'+name+'_ConvBNScaleReLU_Scale, name=name+\'_ConvBNScaleReLU_ReLU\', in_place=True)')
    exec('net.'+name+'_ConvBNScaleReLU_Dropout = L.Dropout(net.'+name+'_ConvBNScaleReLU_ReLU, name=name+\'_ConvBNScaleReLU_Dropout\', dropout_ratio=dropout_ratio)')
    exec('net.ConvBNScaleReLU = net.'+name+'_ConvBNScaleReLU_Dropout')
    return net.ConvBNScaleReLU

def SConvBlockV2(net, from_layer, name, num_input, num_output, channel_multiper=1, kernel_size=3, stride=1, pad=1, dropout_ratio=0.2):
    exec('net.'+name+'_SConvBlockV2_T1  = L.SparseConvolution(from_layer, name=name+\'_SConv_SparseConv\', channel_expand_multiplier=channel_multiper, weight_filler=dict(type=\'xavier\') )')
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
        'moving_average_fraction': 0.999,
        }
    exec('net.'+name+'_SConvBlockV2_BN      = L.BatchNorm(net.'+name+'_SConvBlockV2_T1, name=name+\'_SConv_BN\', in_place=True, **bn_kwargs)')
    exec('net.'+name+'_SConvBlockV2_Scale   = L.Scale(net.'+name+'_SConvBlockV2_BN, name=name+\'_SConv_Scale\', bias_term=True)')
    exec('net.'+name+'_SConvBlockV2_Conv1x1 = L.Convolution(net.'+name+'_SConvBlockV2_Scale, name=name+\'_SConv_Conv1x1\', kernel_size=1, stride=1, num_output=num_output, pad=0, bias_term=True, weight_filler=dict(type=\'xavier\'), bias_filler=dict(type=\'constant\'))')
    exec('net.'+name+'_SConvBlockV2_ReLU    = L.ReLU(net.'+name+'_SConvBlockV2_Conv1x1, name=name+\'_SConv_ReLU\', in_place=True)')
    exec('net.'+name+'_SConvBlockV2_Dropout = L.Dropout(net.'+name+'_SConvBlockV2_ReLU, name=name+\'_SConv_Dropout\', dropout_ratio=dropout_ratio)')
    exec('net.SConvBlockV2 = net.'+name+'_SConvBlockV2_Dropout')
    return net.SConvBlockV2

def VGGNetBody(net, from_layer):
    DropRatio=0.25
    channel_multiper = 4

    #block 1
    from_layer = ConvBNScaleReLU(net, from_layer, name='B0_0', num_output=64, kernel_size=3, stride=1, pad=1, l1_lambda=l1_lambda, dropout_ratio=DropRatio)
    #from_layer = SConvBlockV2(net, from_layer, name='B0_0', num_input=3,  num_output=64, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = SConvBlockV2(net, from_layer, name='B0_1', num_input=64, num_output=64, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = L.Pooling(from_layer, name='B0'+'_Pooling', pool=P.Pooling.MAX, kernel_size=2, pad=0,stride=2)

    #block 2
    from_layer = SConvBlockV2(net, from_layer, name='B1_0', num_input=64, num_output=128, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = SConvBlockV2(net, from_layer, name='B1_1', num_input=128, num_output=128, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = L.Pooling(from_layer, name='B1'+'_Pooling', pool=P.Pooling.MAX, kernel_size=2, pad=0,stride=2)

    #block 3
    from_layer = SConvBlockV2(net, from_layer, name='B2_0', num_input=128, num_output=192, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = SConvBlockV2(net, from_layer, name='B2_1', num_input=192, num_output=192, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = SConvBlockV2(net, from_layer, name='B2_2', num_input=192, num_output=192, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = L.Pooling(from_layer, name='B2'+'_Pooling', pool=P.Pooling.MAX, kernel_size=2, pad=0,stride=2)

    #block 4
    from_layer = SConvBlockV2(net, from_layer, name='B3_0', num_input=192, num_output=192, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = SConvBlockV2(net, from_layer, name='B3_1', num_input=192, num_output=192, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = SConvBlockV2(net, from_layer, name='B3_2', num_input=192, num_output=192, channel_multiper=channel_multiper, kernel_size=3, stride=1, pad=1, dropout_ratio=DropRatio)
    from_layer = L.Pooling(from_layer, name='B3'+'_Pooling', pool=P.Pooling.AVE, global_pooling=True)

    return from_layer

def add_classify_header(net, bottom, classes):
    net.fc1 = L.InnerProduct(bottom, num_output=384, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    net.classifier = L.InnerProduct(net.fc1, num_output=classes, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    return net

def make_VGGNet_deploy():
    deploy_path="deploy.pt"
    net = caffe.NetSpec()
    net.data = L.Input(shape=[dict(dim=[1, num_channels, imgcrop_size, imgcrop_size])])
    net.VGGBody = VGGNetBody(net, net.data)
    add_classify_header(net, net.VGGBody, classes=ClassNum)
    net.prob = L.Softmax(net.classifier)
    with open(deploy_path, 'w') as deploy:
        deploy.write(str(net.to_proto()))
    return

def make_VGGNet_train_test():
    train_test_path="train_test.pt"
    net = caffe.NetSpec()
    net.data, net.label = L.ImageData(name="InputData", source=train_lst, batch_size=train_batch, ntop=2, is_color=True, shuffle=True, transform_param=dict(mirror=True, crop_size=imgcrop_size), include=dict(phase=0)) 
    net.data_test, net.label_test = L.ImageData(name="InputData", source=test_lst, batch_size=test_batch, ntop=2, is_color=True, shuffle=True, transform_param=dict(mirror=True, crop_size=imgcrop_size), include=dict(phase=1))

    #net body
    bottom = VGGNetBody(net, net.data)

    add_classify_header(net, bottom, classes=ClassNum)
    net.softmax_loss = L.SoftmaxWithLoss(net.classifier, net.label)
    net.accuracy = L.Accuracy(net.classifier, net.label, include=dict(phase=1))
    train_test_pt=str(net.to_proto())
    train_test_pt=train_test_pt.replace('data_test', 'data')
    train_test_pt=train_test_pt.replace('label_test', 'label')
    with open(train_test_path, 'w') as train_test:
        train_test.write(train_test_pt)
    return

def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE
    s.net = 'train_test.pt'
    s.display = snapshot/100
    s.test_interval = snapshot/10
    s.test_iter.append(10000)
    s.lr_policy=lr_policy
    s.max_iter = max_step
    if lr_policy=='multistep':
        for step_i in range(len(step_ratio)):
            s.stepvalue.append(int(step_ratio[step_i] * s.max_iter))
    s.snapshot = snapshot
    s.snapshot_prefix = 'weights_'+time_string+'/'+str(argument0)
    s.base_lr = base_lr
    s.momentum = 0.9
    s.weight_decay = global_weight_decay
    s.gamma = 0.1
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    s.type = 'SGD'
    snapshot_format = 'snapshot_format : HDF5'
    solver_path = 'run_'+time_string+'_solver.pt'
    with open(solver_path, 'w') as solver_file:
        solver_file.write(str(s))
    return

def make_run_script():
    log_file='$0.log'
    run_s = 'netname=train_test.pt;\nrm ' + log_file + ';\nnohup ' +  str(caffe_path) + '/build/tools/caffe train --solver=$0_solver.pt --weights=$1 > ' + log_file +' --gpu '+gpu_id+'& \n'
    script_path = 'run_'+time_string
    with open(script_path, 'w') as script_file:
        script_file.write(run_s)
    command = 'chmod +x ' + script_path +'; rm weights_'+time_string+' -rf; mkdir weights_'+time_string+';'
    os.system(command)

if __name__ == '__main__':
    make_VGGNet_deploy()
    make_VGGNet_train_test()
    make_solver()
    make_run_script()
