#ifndef CAFFE_SPARSE_CONVOLUTION_LAYER_HPP_
#define CAFFE_SPARSE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SparseConvolutionLayer : public Layer<Dtype>
{
 public:
  explicit SparseConvolutionLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SparseConvolution"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int InputChannel_; //c
  int OutputChannel_; //c'
  int KH_, KW_; //kh,kw
  int ChannelAxis_;
  int InnerDim_; //h*w
  Blob<Dtype> BottomDataColBuffer_;
  Blob<Dtype> TopDiffColBuffer_;
  Blob<Dtype> ChannelExpandMultipliers_;
  Blob<Dtype> InverseMap_; //[c'], 反向映射
};

}  // namespace caffe

#endif  // CAFFE_SPARSE_CONVOLUTION_LAYER_HPP_
