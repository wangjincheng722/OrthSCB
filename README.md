# OrthSCB
1. Train scripts, logfiles, resulted models for SCNN is in ./SCNN4(8,12)
2. Train scripts, models for Slimming SCNN-4(8,12) is in ./SCNN4(8,12)_Slim, original logfiles is not presented since the size is too large(>25MB).
3. Train scripts, models for orthogonalized SCNN-4(8,12) is in ./SCNN4(8,12)_Orth.
4. Codes for SparseConvolutionLayer are in ./SparseConvolutionLayer/. One might follow the standard configuration with Caffe. Only support CUDA yet.
5. For details, please refer to 
@article{Wang2021,
author = "Jincheng Wang, Lu, Jia",
title = "{Orthogonal sparse convolution module for deep neural networks acceleration}",
year = "2021",
month = "10",
url = "https://www.techrxiv.org/articles/preprint/Orthogonal_sparse_convolution_module_for_deep_neural_networks_acceleration/16830718",
doi = "10.36227/techrxiv.16830718.v1"
}
