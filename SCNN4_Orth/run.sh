netname=/home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/4.TrainRuns/112.BW4SparseConvV9SparseNetV4.5DataAug/PruneTh1.e-2/OrthTh0.3/OrthModel.pt;
rm /home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/4.TrainRuns/112.BW4SparseConvV9SparseNetV4.5DataAug/PruneTh1.e-2/OrthTh0.3/solver.log;
nohup /home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/5.SparseConvCaffeV9.1/build/tools/caffe train --solver=/home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/4.TrainRuns/112.BW4SparseConvV9SparseNetV4.5DataAug/PruneTh1.e-2/OrthTh0.3/solver.pt --weights=/home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/4.TrainRuns/112.BW4SparseConvV9SparseNetV4.5DataAug/PruneTh1.e-2/OrthTh0.3/OrthParam.caffemodel --gpu 0 > /home/jcwang/work/2.GeneratedSparseConvolution/0.CaffeSparseConv/4.TrainRuns/112.BW4SparseConvV9SparseNetV4.5DataAug/PruneTh1.e-2/OrthTh0.3/solver.log&