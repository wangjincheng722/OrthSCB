//By Wang Jincheng 181518052@qq.com
#include <vector>

#include "caffe/layers/sparse_convolution_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;
namespace caffe {

template <typename Dtype>
__global__ void SConvFeat2ColKernel(const Dtype * Feat, Dtype* ColBuff, int KH, int KW,
                               int H, int W, int HW, int CHW, int HWKHKW, int CHWKHKW)
{
  int iN = blockIdx.x;
  int iC = blockIdx.y;
  int iH = threadIdx.x;
  int iW = threadIdx.y;
  int M  = ((KH+1)>>1)-1;
  int N  = ((KW+1)>>1)-1;
  int ColBuffPos = iN*CHWKHKW + iC*HWKHKW + iH*W*KH*KW + iW*KH*KW;
  int FeatPos    = iN*CHW     + iC*HW     + (iH-M)*W   + iW-N;
  for(int iKH=0; iKH<KH; iKH++)
    for(int iKW=0; iKW<KW; iKW++)
    {
      if((iH==0&&iKH<M) || (iW==0&&iKW<N) || (iH==H-1&&iKH>M) || (iW==W-1&&iKW>N))
        ColBuff[ ColBuffPos + iKH*KW + iKW ] = 0;
      else
        ColBuff[ ColBuffPos + iKH*KW + iKW ] = Feat[ FeatPos + iKH*W + iKW ];
    }
}

template <typename Dtype>
void SConvFeat2Col(const Dtype * Feat, Dtype* ColBuff, int N, int C,int H, int W, int KH=3, int KW=3)
{
  int HW      = H*W;
  int CHW     = C*H*W;
  int HWKHKW  = H*W*KH*KW;
  int CHWKHKW = C*H*W*KH*KW;
  dim3 DimGrid(N, C);
  dim3 DimBlock(H, W);
  SConvFeat2ColKernel<<<DimGrid, DimBlock>>>(Feat, ColBuff, KH, KW, H, W, HW, CHW, HWKHKW, CHWKHKW);
}

template <typename Dtype>
__global__ void SConvDataForwardKernel(Dtype * TopData, const Dtype* ColBufferData, const Dtype* KernelData, const Dtype* InverseMapData,
                                  int N, int KHKW, int H, int W, int TopSliceLength, int BottomSliceLength, int HW)
{
  int iN   = blockIdx.x;
  int iCP  = blockIdx.y;
  int iH   = blockIdx.z;
  int iW   = threadIdx.x;
  int iC   = (int)InverseMapData[iCP];
  int TopDataPos       = iN*TopSliceLength+iCP*HW+iH*W+iW;
  int ColBufferDataPos = iN*BottomSliceLength*KHKW+iC*HW*KHKW+iH*W*KHKW+iW*KHKW;
  int KernelDataPos    = iCP*KHKW;
  for(int iKHKW=0; iKHKW<KHKW; iKHKW++)
    TopData[TopDataPos] += ColBufferData[ColBufferDataPos+iKHKW]*KernelData[KernelDataPos+iKHKW];
}

template <typename Dtype>
void SparseConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const Dtype* BottomData = bottom[0]->gpu_data();
  const Dtype* KernelData = this->blobs_[0]->gpu_data();
  Dtype* TopData          = top[0]->mutable_gpu_data();
  Dtype* BottomDataColBufferData = BottomDataColBuffer_.mutable_gpu_data();
  const Dtype* InverseMapData    = InverseMap_.gpu_data();
  int N                   = top[0]->shape()[0];
  int C                   = InputChannel_;
  int CP                  = OutputChannel_;
  int H                   = top[0]->shape()[ChannelAxis_+1];
  int W                   = top[0]->shape()[ChannelAxis_+2];
  int HW                  = InnerDim_;
  int TopSliceLength      = top[0]->count(ChannelAxis_);
  int BottomSliceLength   = bottom[0]->count(ChannelAxis_);

  //1st top forward
  caffe_gpu_set(top[0]->count(0), (Dtype)0, TopData);
  SConvFeat2Col(BottomData, BottomDataColBufferData, N, C, H, W, KH_, KW_);
  dim3 DimGrid(N, CP, H); 
  dim3 DimBlock(W);
  SConvDataForwardKernel<<<DimGrid, DimBlock>>>(TopData, BottomDataColBufferData, KernelData, InverseMapData,
                                                N, KH_*KW_, H, W, TopSliceLength, BottomSliceLength, HW);
}

//debug: problem here
template <typename Dtype>
__global__ void SConvDiffBackwardKernel(Dtype * BottomDiff, const Dtype* TopDiffColBufferData, const Dtype* KernelData, const Dtype* MapData,
                                        int N, int C, int H, int W, int KH, int KW, 
                                        int TopSliceLength, int BottomSliceLength, int HW, int KHKW)
{
  int iN = blockIdx.x;
  int iC = blockIdx.y;
  int iH = blockIdx.z;
  int iW = threadIdx.x;
  int EC = (int)MapData[iC]; //expanded channels
  int ChBase = 0;
  for(int iIC=0; iIC<iC; iIC++)
    ChBase += (int)MapData[iIC];
  
  int BottomDiffPos = iN*BottomSliceLength+iC*HW+iH*W+iW;
  int TopDiffPos0   = iN*TopSliceLength*KHKW;
  for(int iEC=0; iEC<EC; iEC++)
  {
    int iCP = ChBase + iEC;
    int TopDiffPos1   = TopDiffPos0 + iCP*HW*KHKW + (iH*W+iW)*KHKW;
    int KernelDataPos = (iCP+1)*KHKW - 1; //Rotated by 180 degree
    for(int iKHKW=0; iKHKW<KHKW; iKHKW++)
      BottomDiff[BottomDiffPos] += TopDiffColBufferData[TopDiffPos1+iKHKW]*KernelData[KernelDataPos-iKHKW];
  }
}

template <typename Dtype>
__global__ void SConvConvKernelUpdate(Dtype * KernelDiff, const Dtype* TopDiff, const Dtype* BottomDataColBuffer, const Dtype* InverseMapData,
                                      int N, int CP, int CPB, int H, int W, int KH, int KW, 
                                      int TopSliceLength, int BottomSliceLengthKHKW, int HW, int KHKW)
{
  int iCPN  = blockIdx.x;
  int iKHKW = threadIdx.x;
  int iCPB  = threadIdx.y;
  int iCP   = iCPN*CPB + iCPB;
  if(iCP < CP)
  {
    int KernelPos = iCP*KHKW+iKHKW;
    int iC = (int)InverseMapData[iCP];
    for(int iHW=0; iHW<HW; iHW++)
    {
      int TopDiffPos             = iCP*HW+iHW;
      int BottomDataColBufferPos = iC*HW*KHKW + iHW*KHKW + iKHKW;
      for(int iN=0; iN<N; iN++)
      {
        KernelDiff[KernelPos]   += TopDiff[TopDiffPos + iN*TopSliceLength] * BottomDataColBuffer[BottomDataColBufferPos + iN*BottomSliceLengthKHKW];
      }
    }
  }
}

template <typename Dtype>
void SparseConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  if ( !(this->param_propagate_down_[0] || propagate_down[0]) ) { return; }

  const Dtype* TopDiff                 = top[0]->gpu_diff();
  Dtype* TopDiffColBufferData          = TopDiffColBuffer_.mutable_gpu_diff();
  const Dtype* BottomData              = bottom[0]->gpu_data();
  Dtype* BottomDataColBufferData       = BottomDataColBuffer_.mutable_gpu_data();
  Dtype* BottomDiff                    = bottom[0]->mutable_gpu_diff();
  const Dtype* KernelData              = this->blobs_[0]->gpu_data();
  Dtype* KernelDiff                    = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* InverseMapData          = InverseMap_.gpu_data();
  const Dtype* MapData                 = ChannelExpandMultipliers_.gpu_data();

  int N                                = top[0]->shape()[0];
  int H                                = top[0]->shape()[ChannelAxis_+1];
  int W                                = top[0]->shape()[ChannelAxis_+2];
  int CP                               = OutputChannel_;
  int C                                = InputChannel_;
  int TopSliceLength                   = top[0]->count(ChannelAxis_);
  int BottomSliceLength                = bottom[0]->count(ChannelAxis_);
  int HW                               = InnerDim_;
  int KHKW                             = KH_*KW_;
  int BottomSliceLengthKHKW            = BottomSliceLength*KHKW;

  //1st top diff. back propagation
  //Directly by cuda
  
  if (propagate_down[0])
  {
    caffe_gpu_set(bottom[0]->count(0), (Dtype)0, BottomDiff);
    SConvFeat2Col(TopDiff, TopDiffColBufferData, N, CP, H, W, KH_, KW_);
    dim3 DimGrid(N, C, H);
    dim3 DimBlock(W);
    SConvDiffBackwardKernel<<<DimGrid, DimBlock>>>(BottomDiff, TopDiffColBufferData, KernelData, MapData, 
                                                   N, C, H, W, KH_, KW_, 
                                                   TopSliceLength, BottomSliceLength, HW, KHKW);
  }
  
  if (this->param_propagate_down_[0])
  {
    //1st top diff. to update convolution kernel
    caffe_gpu_set(this->blobs_[0]->count(0), (Dtype)0, KernelDiff);
    SConvFeat2Col(BottomData, BottomDataColBufferData, N, C, H, W, KH_, KW_);
    int CPB = 64;
    int CPN = CP/CPB + 1;
    dim3 DimGrid2(CPN);
    dim3 DimBlock2(KHKW,CPB);
    SConvConvKernelUpdate<<<DimGrid2, DimBlock2>>>(KernelDiff, TopDiff, BottomDataColBufferData, InverseMapData,
                                                   N, CP, CPB, H, W, KH_, KW_, 
                                                   TopSliceLength, BottomSliceLengthKHKW, HW, KHKW);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseConvolutionLayer);
}  // namespace caffe
