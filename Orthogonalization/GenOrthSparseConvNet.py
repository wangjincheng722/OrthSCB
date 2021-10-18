#!/usr/bin/env python2
import math, sys, os, copy, shutil, struct
os.environ['GLOG_minloglevel']='3'
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
from GenerateFineTuneScripts import make_solver, make_run_script
from SimulateOrthonormalBasis import *

LayerEnds             = "SparseConv"
DownSamplingLayerEnds = "Pooling"
DataShape             = [32,32]
def GetLayerWeights(ModelPath, WeightsPath, LayerNamePrefix=""):
    net=caffe.Net(ModelPath,WeightsPath,caffe.TEST)
    AllLayerWeights=[]
    for param_name in net.params.keys():
        Weights=[]
        if param_name[0:len(LayerNamePrefix)] == LayerNamePrefix:
            Weights.append(param_name)
            for Slice in net.params[param_name]:
                weight=Slice.data
                Weights.append(weight)
            AllLayerWeights.append(Weights)
    return AllLayerWeights

def GetConvLayerInputShape(ModelPath, ConvLayerNameEnd = "SparseConv", DownSamplingLayerEnds = "Pooling"):
    ConvInputShapeDict = {"Data":DataShape}

    net = caffe_pb2.NetParameter()
    text_format.Merge(open(ModelPath).read(), net)
    LayerNames = []
    for Layer in net.layer:
        LayerNames.append(Layer.name)

    CurrInputShape     = copy.deepcopy(DataShape)
    for LayerName in LayerNames:
        if LayerName.endswith(DownSamplingLayerEnds):
            for i in range(len(CurrInputShape)):
                CurrInputShape[i] = CurrInputShape[i]/2
        if LayerName.endswith(ConvLayerNameEnd):
            ConvInputShapeDict[LayerName] = copy.deepcopy(CurrInputShape)
    print("[I]@GetConvLayerInputShape: ConvInputShapeDict={}".format(ConvInputShapeDict))
    return ConvInputShapeDict

def OrthSparseConvolution(SparseConvParams, BNParams, ScaleParams, Conv1x1Params, OrthBasisArr, TransMatArr, InputShape):
    #Notice the sparse convolution consists of 4 terms which need to be pruned carefully:
    #(1) SparseConvolutionLayer. The pruned sparse matrix should be saved
    #    as a binary file as the initial input. The blobs_[0](stored in LayerWeights[1]) is the weights with 
    #    shape [ec, kh, kw]. The pruned elements in sparse matrix are related to the kernels respectively. Once
    #    a element is pruned, the corresponding kernel is dropped.
    #(2) BatchNormLayer. Should be pruned.
    #(3) ScaleLayer. Bias=False. Should be pruned.
    #(4) Conv1x1Layer. Bias=True
    #len(SparseExpandParam)=CP
    print("[I]@OrthSparseConvolution: Src. from "+SparseConvParams[0]+" shape "+str(SparseConvParams[1].shape))
    print("[I]@OrthSparseConvolution: Src. from "+BNParams[0]+" shape "+str(BNParams[1].shape)+", "+str(BNParams[2].shape)+", "+str(BNParams[3].shape) )
    print("[I]@OrthSparseConvolution: Src. from "+ScaleParams[0]+" shape "+str(ScaleParams[1].shape))
    print("[I]@OrthSparseConvolution: Src. from "+Conv1x1Params[0]+" shape "+str(Conv1x1Params[1].shape)+", "+str(Conv1x1Params[2].shape))
    #Orth. sparse conv. params
    OldTotalChannels = SparseConvParams[1].shape[0]
    OrthSparseConvParams = []
    OrthSparseConvParams.append(copy.deepcopy(SparseConvParams[0])) #1st item is the layer name
    C             = len(OrthBasisArr)
    KHKW          = OrthBasisArr[0].shape[1]
    KH            = SparseConvParams[1].shape[1]
    KW            = SparseConvParams[1].shape[2]
    FeatH         = InputShape[0]
    FeatW         = InputShape[1]
    OrthKernelNum = 0
    for i in range(C):
        OrthKernelNum += OrthBasisArr[i].shape[0]
    OrthSparseConv = np.zeros((OrthKernelNum, KH, KW), dtype='float')
    print("[I]@OrthSparseConvolution: OrthSparseConv.shape={}, OrthBasisArr.length={}".format(OrthSparseConv.shape, len(OrthBasisArr)))
    Ctr = 0
    for i in range(C):
        M = OrthBasisArr[i].shape[0]
        for m in range(M):
            OrthSparseConv[Ctr] = copy.deepcopy(OrthBasisArr[i][m]).reshape(KH,KW)
            Ctr += 1
    OrthSparseConvParams.append(OrthSparseConv) #Kernels

    #Conv. 1x1
    OrthConv1x1Params = []
    OrthConv1x1Params.append(copy.deepcopy(Conv1x1Params[0])) #1st item is the layer name
    CP                = Conv1x1Params[1].shape[0]
    OrthConv1x1       = np.zeros((CP, OrthKernelNum, 1, 1), dtype='float')
    print("[I]@OrthSparseConvolution: OrthConv1x1.shape={}".format(OrthConv1x1.shape))
    #Cal. k^l,1x1_jih
    Kl1x1 = []
    for j in range(CP):
        Kl1x1.append([])
        Ctr_ih = 0
        for i in range(C):
            Kl1x1[j].append([])
            H = TransMatArr[i].shape[0]
            for h in range(H):
                Kl1x1[j][i].append(Conv1x1Params[1][j][Ctr_ih][0][0])
                Ctr_ih += 1
    #Cal. alpha_ih and beta_ih
    #TBFixed
    Alpha = []
    Beta  = []
    Ctr   = 0
    Eps   = 1.e-3
    Frac  = 1./float(BNParams[3][0])
    Mean  = BNParams[1]*Frac
    Var   = BNParams[2]*Frac
    Gamma = ScaleParams[1]
    ScaleBias = ScaleParams[2]
    for i in range(C):
        Alpha.append([])
        Beta.append([])
        H = TransMatArr[i].shape[0]
        for h in range(H):
            #alpha = gamma/sqrt(var+eps)
            AlphaValue =  Gamma[Ctr]/np.sqrt(Var[Ctr]+Eps)
            #beta  = -gamma*mean/sqrt(var+eps)
            #BetaValue  =  -Gamma[Ctr]*Mean[Ctr]/np.sqrt(Var[Ctr]+Eps)
            BetaValue  =  -Gamma[Ctr]*Mean[Ctr]/np.sqrt(Var[Ctr]+Eps) + ScaleBias[Ctr]
            #print("[D]@OrthSparseConvolution: Mean[{0}]={1}, Var[{0}]={2}, Gamma[{0}]={3}, Alpha[{0}]={4}, Beta[{0}]={5}".format(Ctr, Mean[Ctr], Var[Ctr], Gamma[Ctr], AlphaValue, BetaValue))
            Alpha[i].append(AlphaValue)
            Beta[i].append(BetaValue)
            Ctr += 1
    #Cal. kTilde^l,1x1_jim
    KTildel1x1 = []
    for j in range(CP):
        KTildel1x1.append([])
        for i in range(C):
            KTildel1x1[j].append([])
            H = TransMatArr[i].shape[0]
            M = TransMatArr[i].shape[1]
            for m in range(M):
                KTildel1x1Value = 0
                for h in range(H):
                    KTildel1x1Value += Kl1x1[j][i][h]*Alpha[i][h]*TransMatArr[i][h][m]
                KTildel1x1[j][i].append(KTildel1x1Value)
    #Cal. bias
    Bias = []
    for j in range(CP):
        BaisValue = Conv1x1Params[2][j]
        for i in range(C):
            H = TransMatArr[i].shape[0]
            for h in range(H):
                BaisValue += Kl1x1[j][i][h]*Beta[i][h]
        Bias.append(BaisValue)
    #rearange data
    R_ih = 0
    R_im = 0
    for i in range(C):
        R_ih += TransMatArr[i].shape[0]
        R_im += TransMatArr[i].shape[1]
    Conv1x1ParamsKer = np.zeros((CP,R_im,1,1), dtype='float')
    for j in range(CP):
        Ctr_im = 0
        for i in range(C):
            M = TransMatArr[i].shape[1]
            for m in range(M):
              Conv1x1ParamsKer[j][Ctr_im][0][0] = KTildel1x1[j][i][m]
              Ctr_im += 1
    Conv1x1ParamsBias = np.zeros((CP), dtype='float')
    for j in range(CP):
        Conv1x1ParamsBias[j] = Bias[j]
    OrthConv1x1Params.append(copy.deepcopy(Conv1x1ParamsKer)) #Convolution kernel
    OrthConv1x1Params.append(copy.deepcopy(Conv1x1ParamsBias)) #Convolution bias

    #SparseExpandParam
    SparseExpandParam = np.zeros(C)
    for i in range(C):
        SparseExpandParam[i] = TransMatArr[i].shape[1]

    #PruningRatio
    PruningRatio = 1.-float(R_im)/float(R_ih)
    print("[I]@OrthSparseConvolution:SparseExpandParam={}".format(map(int,SparseExpandParam)))
    NewKernelNum = sum(map(int,SparseExpandParam))
    KernelNum = [NewKernelNum,                                    OldTotalChannels,                                    CP*C                  ]
    ParamNum  = [NewKernelNum*(KH*KW+CP+1),                       OldTotalChannels*(KH*KW+CP+1),                       CP*C*KH*KW            ]
    MACs      = [NewKernelNum*(KW*KH*FeatH*FeatW+CP*FeatH*FeatW), OldTotalChannels*(KW*KH*FeatH*FeatW+CP*FeatH*FeatW), CP*C*KH*KW*FeatH*FeatW]
    print("[I]@OrthSparseConvolution:CP={}, C={}, H={}, W={}, KH={}, KW={}".format(CP,C,FeatH,FeatW,KH,KW))
    print("[I]@OrthSparseConvolution:KernelNum={}, ParamNum={}, MACs={}".format(KernelNum, ParamNum, MACs))
    print("[I]@OrthSparseConvolution:PruningRatio={}".format(PruningRatio))
    return OrthSparseConvParams, OrthConv1x1Params, SparseExpandParam, KernelNum, ParamNum, MACs, PruningRatio

def DeleteMakeDir(Path):
    if os.path.exists(Path):
        shutil.rmtree(Path)
    os.makedirs(Path)
    return

def SaveOrthParam(OrthModel, OrthParamPath, AllLayerWeights):
    OrthNet = caffe.Net(OrthModel, caffe.TEST)
    for j in range(len(AllLayerWeights)):
        param_name = AllLayerWeights[j][0]
        for i in range(len(OrthNet.params[param_name])):
            print("[D]@SaveParam: layer name={}, Length={}, shape={}".format(AllLayerWeights[j][0], len(AllLayerWeights[j])-1, AllLayerWeights[j][i+1].shape))
            OrthNet.params[param_name][i].data[:]=AllLayerWeights[j][i+1][:]
    OrthNet.save(OrthParamPath)
    return

def SaveSparseExpandParam(SparseExpandParam, SparseExpandParamPath):
    C = SparseExpandParam.shape[0]
    File  = open(SparseExpandParamPath,'wb')
    #Write input channel number
    Value = struct.pack('f',float(C))
    File.write(Value)
    for iC in range(C):
        Value = struct.pack('f', SparseExpandParam[iC])
        File.write(Value)
    File.close()
    print("[I]@SaveSparseExpandParam: shape ["+str(C)+"] -> "+SparseExpandParamPath)
    return

def GenerateOrthModel(SrcModel, DstModel, SparseExpandParamList):
    # Load source model
    Model = caffe_pb2.NetParameter()
    with open(SrcModel) as File:
        s = File.read()
        text_format.Merge(s, Model)
    File.close()

    # Search and modify SparseConvolution parameter
    Model.name = "OrthSparseConvolutionNet"
    for Element in SparseExpandParamList:
        SparseConvLayerName = Element[0]
        for iL in range(len(Model.layer)):
            Layer = Model.layer[iL]
            if Layer.name==SparseConvLayerName:
                Layer.sparse_convolution_param.channel_expand_file = os.path.abspath(Element[1])
                #fix the conv1x1 bottoms
                Model.layer[iL+3].bottom[0] = Model.layer[iL].top[0]
                #remove following BN and Scale layers
                del Model.layer[iL+1:iL+3]
                break

    # Save pruned model
    with open(DstModel, 'w') as File:
        File.write(str(Model))
    File.close()
    print("[I]@GenerateOrthModel: Orth. sparse conv. model saved to "+str(DstModel))
    return

def GetSparseExpandParam(LayerName, Model, C):
    SparseExpandParam = np.zeros(C)
    for Layer in Model.layer:
        if Layer.name==LayerName:
            if Layer.sparse_convolution_param.channel_expand_file =="":
                SparseExpandParam[:]=Layer.sparse_convolution_param.channel_expand_multiplier
            else:
                File  = open(Layer.sparse_convolution_param.channel_expand_file,'rb')
                FileC = int(struct.unpack('f', File.read(4)))
                if FileC!=C:
                    print("[E]: FileC != C. ("+str(FileC)+" vs. "+str(C)+"). File: "+Layer.sparse_convolution_param.channel_expand_file)
                for iC in range(C):
                    SparseExpandParam[iC] = struct.unpack('f', File.read(4))
                File.close()
    return SparseExpandParam

def OrthSparseNet(ModelPath, DeployPath, WeightsPath, DstPath, PruneTh, OrthTh, SparseConvolutionLayersNameList, ConvLayerInputShapeDict):
    #Load Model
    Model = caffe_pb2.NetParameter()
    with open(ModelPath) as File:
        s = File.read()
        text_format.Merge(s, Model)
    File.close()

    #Get all layer weights
    print("[I]@OrthSparseNet: GetLayerWeights...")
    AllLayerWeights  = GetLayerWeights(ModelPath=ModelPath, WeightsPath=WeightsPath, LayerNamePrefix="")
    OrthParams = copy.deepcopy(AllLayerWeights)
    print("[I]@OrthSparseNet: Done. AllLayerWeights length is "+str(len(AllLayerWeights)))

    #Get sparse conv layer weights
    AllSparseConvLayerWeights = GetSparseConvLayerWeights(ModelPath, WeightsPath)

    #Calculate orthonormal basis array and transformation matrix array
    OrthBasis = []
    for i in range(len(AllSparseConvLayerWeights)): 
        OrthonormalBasisArr, TransMatArr = OrthogonalizationV2(AllSparseConvLayerWeights[i][1], AllSparseConvLayerWeights[i][2], Norm=True, PruneTh=PruneTh, OrthTh=OrthTh) #Expand coefficients and trained weights
        OrthBasis.append([AllSparseConvLayerWeights[i][0], OrthonormalBasisArr, TransMatArr])

    #Delete and make path
    DeleteMakeDir(DstPath)

    #Prune weights
    SparseExpandParamList = []
    AllKernelNum = [0,0,0]
    AllParamNum  = [0,0,0]
    AllMACs      = [0,0,0]
    for i in range(len(AllLayerWeights)):
        # check target layer list
        for SparseConvolutionLayersName in SparseConvolutionLayersNameList:
            if AllLayerWeights[i][0]==SparseConvolutionLayersName:
                SparseConvParams = AllLayerWeights[i]
                BNParams         = AllLayerWeights[i+1]
                ScaleParams      = AllLayerWeights[i+2]
                Conv1x1Params    = AllLayerWeights[i+3]

                for j in range(len(OrthBasis)):
                    if OrthBasis[j][0] == AllLayerWeights[i][0]:
                        OrthBasisArr = OrthBasis[j][1]
                        TransMatArr  = OrthBasis[j][2]

                OrthSparseConvParams, OrthConv1x1Params, SparseExpandParam, KernelNum, ParamNum, MACs, PruningRatio = OrthSparseConvolution(SparseConvParams=SparseConvParams, BNParams=BNParams, ScaleParams=ScaleParams, Conv1x1Params=Conv1x1Params, OrthBasisArr=OrthBasisArr, TransMatArr=TransMatArr, InputShape=ConvLayerInputShapeDict[SparseConvolutionLayersName])
                for j in range(len(AllKernelNum)):
                    AllKernelNum[j] += KernelNum[j]
                    AllParamNum[j]  += ParamNum[j]
                    AllMACs[j]      += MACs[j]
                OrthParams[i]   = OrthSparseConvParams
                OrthParams[i+1] = []
                OrthParams[i+2] = []
                OrthParams[i+3] = OrthConv1x1Params
                SparseExpandParamPath = DstPath+"/"+OrthParams[i][0]+".SparseExpandParam.bin"
                SaveSparseExpandParam(SparseExpandParam=SparseExpandParam, SparseExpandParamPath=SparseExpandParamPath)
                #Save orth. sparse expand param. file
                SparseExpandParamList.append([AllLayerWeights[i][0], SparseExpandParamPath])
    #Remove empty Layers
    Ctr = 0
    for i in range(len(OrthParams)):
        if len(OrthParams[Ctr])==0:
            del OrthParams[Ctr]
        else:
            Ctr += 1

    #Generate and save orth. model
    OrthModelPath  = DstPath+"/OrthModel.pt"
    OrthDeployPath = DstPath+"/OrthDeploy.pt"
    GenerateOrthModel(ModelPath,  OrthModelPath, SparseExpandParamList)
    GenerateOrthModel(DeployPath, OrthDeployPath, SparseExpandParamList)

    #Save model parameter file
    SaveOrthParam(OrthModel=OrthModelPath, OrthParamPath=DstPath+"/OrthParam.caffemodel", AllLayerWeights=OrthParams)
    print("[I]@OrthSparseNet: KernelNum={}, ParamNum={}, MACs={}".format(AllKernelNum, AllParamNum, AllMACs))
    for i in range(len(AllKernelNum)):
        AllKernelNum[i] += 3*64
        AllParamNum[i]  += 3*64*3*3+64 + 384*(192+1) + 384*(10+1)
        AllMACs[i]      += 64*(3*32*32+1)*3*3 + 384*(192+1) + 384*(10+1)
    print("[I]@OrthSparseNet: AllKernelNum={}, AllParamNum={}, AllMACs={}".format(AllKernelNum, AllParamNum, AllMACs))
    print("[I]@OrthSparseNet: Ratio of Prune: ParamNumRatio={}, MACsRatio={}".format(float(AllParamNum[0])/float(AllParamNum[1]), float(AllMACs[0])/float(AllMACs[1])))
    print("[I]@OrthSparseNet: Ratio to Conv.: ParamNumRatio={}, MACsRatio={}".format(float(AllParamNum[0])/float(AllParamNum[2]), float(AllMACs[0])/float(AllMACs[2])))
    return

def GenerateTask(TaskPath, LR, Steps, GlobalWeightDecay, GPUID):
    FTOutputWeightPath = os.path.abspath(TaskPath+"/FTWeights")
    DeleteMakeDir(FTOutputWeightPath)

    ModelPath         = os.path.abspath(TaskPath+"/OrthModel.pt")
    FTInputWeightPath = os.path.abspath(TaskPath+"/OrthParam.caffemodel")
    SolverPath        = os.path.abspath(TaskPath+"/solver.pt")

    make_solver(SolverPath=SolverPath, ModelPath=ModelPath, MaxStep=Steps, BaseLR=LR, GlobalWeightDecay=GlobalWeightDecay, TrainWeightPath=FTOutputWeightPath)
    make_run_script(TaskPath = TaskPath, ModelPath=ModelPath, ModelParamPath=FTInputWeightPath, GPUID=GPUID)
    return

def GetLayerNames(ModelPath):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(ModelPath).read(), net)
    LayerNames = []
    for Layer in net.layer:
        LayerNames.append(Layer.name)
    print("[I]@GetLayerNames: All layer names are " + str(LayerNames))
    return LayerNames

def CreateOrthTrainJob(ModelPath, DeployPath, ParaPath, TaskPath, LR, Steps, GlobalWeightDecay, PruneTh, OrthTh, GPUID):
    #Get target layers
    LayerNames = GetLayerNames(ModelPath = os.path.abspath(ModelPath))
    SparseConvolutionLayersNameList = []
    for LayerName in LayerNames:
        if LayerName.endswith(LayerEnds):
            SparseConvolutionLayersNameList.append(LayerName)
    print("[I]@main: SparseConvolution layers are " + str(SparseConvolutionLayersNameList))
    if len(SparseConvolutionLayersNameList)==0:
        print("[E]: No target layers.")
        exit(0)

    #Get ConvLayerInputShapeDict
    ConvLayerInputShapeDict = GetConvLayerInputShape(ModelPath, ConvLayerNameEnd = LayerEnds, DownSamplingLayerEnds = DownSamplingLayerEnds)

    #Prune
    OrthSparseNet(ModelPath                  = os.path.abspath(ModelPath), 
                  DeployPath                 = os.path.abspath(DeployPath),
                  WeightsPath                = os.path.abspath(ParaPath), 
                  DstPath                    = os.path.abspath(TaskPath),
                  PruneTh                    = PruneTh,
                  OrthTh                     = OrthTh,
                  SparseConvolutionLayersNameList = SparseConvolutionLayersNameList,
                  ConvLayerInputShapeDict    = ConvLayerInputShapeDict
                  )

    #Generate finetune scripts
    GenerateTask(os.path.abspath(TaskPath), LR, Steps, GlobalWeightDecay, GPUID=GPUID)

    #Run Job
    RunPath = os.path.abspath(TaskPath)+'/run.sh'
    #os.system(RunPath)

    OrthModelPath  = os.path.abspath(TaskPath+"/OrthModel.pt")
    OrthDeployPath = os.path.abspath(TaskPath+"/OrthedDeploy.pt")
    OrthParaPath   = os.path.abspath(TaskPath+"/FTWeights/FT_iter_"+str(Steps)+".caffemodel")
    return OrthModelPath, OrthDeployPath, OrthParaPath
