#include <algorithm>
#include <vector>
#include <fstream>

#include "caffe/filler.hpp"
#include "caffe/layers/sparse_convolution_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;
namespace caffe {

template <typename Dtype>
void SparseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const SparseConvolutionParameter& sparse_convolution_param = this->layer_param_.sparse_convolution_param();
  ChannelAxis_   = sparse_convolution_param.channel_axis(); //ChannelAxis_
  InputChannel_  = bottom[0]->shape(ChannelAxis_);     //c
  KH_ = sparse_convolution_param.kernel_h(); //kh
  KW_ = sparse_convolution_param.kernel_w(); //kw

  // Initialize OutputChannel_
  string ChannelExpandParamFile = this->layer_param_.sparse_convolution_param().channel_expand_file(); //channel_expand_file
  int ChannelExpandMultiplier = this->layer_param_.sparse_convolution_param().channel_expand_multiplier();
  OutputChannel_ = 0;
  if (ChannelExpandParamFile != "")
  {
    ifstream Fin(ChannelExpandParamFile.c_str(), ios::in | ios::binary);
    Dtype InputChannelToExpand;
    Fin.read((char*)&(InputChannelToExpand), sizeof(Dtype));
    CHECK_EQ(int(InputChannelToExpand), InputChannel_) << "The channel in the "<<ChannelExpandParamFile<<" should == InputChannel_ ("<<int(InputChannelToExpand)<<" vs. "<<InputChannel_<<")";
    Dtype ChannelExpandMultiplierTMP;
    for(int i=0; i<InputChannel_; i++)
    {
      Fin.read((char*)&(ChannelExpandMultiplierTMP), sizeof(Dtype));
      OutputChannel_ += (int)ChannelExpandMultiplierTMP;
    }
    Fin.close();
  }
  else
    OutputChannel_ = InputChannel_*ChannelExpandMultiplier;

  //check this->blob[0] - convolution kernels  
  if (this->blobs_.size() > 0 )
  {
    //check blob number
    CHECK_EQ(1, this->blobs_.size()) << "Incorrect number of weight blobs.";
    //check blob[0] shape
    if (OutputChannel_ != this->blobs_[0].get()->shape()[0]) 
      LOG(FATAL) << "Incorrect Sparse convolution shape: expected shape "  << OutputChannel_ << "; instead, shape was " << this->blobs_[0]->shape_string();
  }
  
  this->blobs_.resize(1);
  vector<int> KernelShape; //[c', kh, kw]
  KernelShape.push_back(OutputChannel_);
  KernelShape.push_back(KH_);
  KernelShape.push_back(KW_);
  this->blobs_[0].reset(new Blob<Dtype>(KernelShape));
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(sparse_convolution_param.weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  //初始化InnerDim_
  InnerDim_ = bottom[0]->count(ChannelAxis_+1); //h*w
}

//在此函数中reshape， Blob在内存和显存中会始终存在
template <typename Dtype>
void SparseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  //Reshape top
  CHECK_EQ(top.size(), 1) << this->type() << " Layer has 1 top.";
  //Reshape 1st top
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not allow in-place computation.";
  vector<int> TopShape(bottom[0]->shape());
  TopShape[ChannelAxis_] = OutputChannel_;
  top[0]->Reshape(TopShape); //[n, c', h, w]

  // Initialize ChannelExpandMultipliers_
  string ChannelExpandParamFile = this->layer_param_.sparse_convolution_param().channel_expand_file(); //channel_expand_file
  int ChannelExpandMultiplier = this->layer_param_.sparse_convolution_param().channel_expand_multiplier();
  if (ChannelExpandParamFile != "")
  {
    //ChannelExpandMultipliers_
    vector<int> ChannelExpandMultipliersShape(1);
    ChannelExpandMultipliersShape[0] = InputChannel_;
    ChannelExpandMultipliers_.Reshape(ChannelExpandMultipliersShape);
    ifstream Fin(ChannelExpandParamFile.c_str(), ios::in | ios::binary);
    Dtype InputChannelToExpand;
    Fin.read((char*)&(InputChannelToExpand), sizeof(Dtype));
    for(int i=0; i<InputChannel_; i++)
    {
      Fin.read((char*)&(ChannelExpandMultipliers_.mutable_cpu_data()[i]), sizeof(Dtype));
    }
    Fin.close();
  }
  else
  {
    //ChannelExpandMultipliers_
    vector<int> ChannelExpandMultipliersShape(1);
    ChannelExpandMultipliersShape[0] = InputChannel_;
    ChannelExpandMultipliers_.Reshape(ChannelExpandMultipliersShape);
    caffe_set(ChannelExpandMultipliers_.count(0), (Dtype)ChannelExpandMultiplier, ChannelExpandMultipliers_.mutable_cpu_data());
  }

  //Reshape the BottomDataColBuffer_
  vector<int> BottomDataColBufferShape;
  for(size_t i=0;i<bottom[0]->shape().size();i++)
    BottomDataColBufferShape.push_back(bottom[0]->shape()[i]);
  BottomDataColBufferShape.push_back(KH_);
  BottomDataColBufferShape.push_back(KW_);
  BottomDataColBuffer_.Reshape(BottomDataColBufferShape);

  //Reshape the TopDiffColBuffer_
  vector<int> TopDiffColBufferShape;
  for(size_t i=0;i<top[0]->shape().size();i++)
    TopDiffColBufferShape.push_back(top[0]->shape()[i]);
  TopDiffColBufferShape.push_back(KH_);
  TopDiffColBufferShape.push_back(KW_);
  TopDiffColBuffer_.Reshape(TopDiffColBufferShape);
  
  //initialize InverseMap_
  vector<int> InverseMapShape;
  InverseMapShape.push_back(OutputChannel_);
  InverseMap_.Reshape(InverseMapShape);
  int Ctr=0;
  for(int i=0; i<InputChannel_; i++)
    for(int j=0; j<(int)ChannelExpandMultipliers_.cpu_data()[i]; j++)
    {
      InverseMap_.mutable_cpu_data()[Ctr] = (Dtype)i;
      Ctr++;
    }
}

template <typename Dtype>
void SparseConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void SparseConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
}


#ifdef CPU_ONLY
STUB_GPU(SparseConvolutionLayer);
#endif

INSTANTIATE_CLASS(SparseConvolutionLayer);
REGISTER_LAYER_CLASS(SparseConvolution);

}  // namespace caffe
