netname=/home/jcwang/work/0.SparseConv/4.Train/111.BW8SparseConvV9.1SparseNetV4.5/PruneTh1.e-2/OrthTh0.3/OrthModel.pt;
rm /home/jcwang/work/0.SparseConv/4.Train/111.BW8SparseConvV9.1SparseNetV4.5/PruneTh1.e-2/OrthTh0.3/solver.log;
nohup /home/jcwang/work/0.SparseConv/0.SparseCaffeV9.1/build/tools/caffe train --solver=/home/jcwang/work/0.SparseConv/4.Train/111.BW8SparseConvV9.1SparseNetV4.5/PruneTh1.e-2/OrthTh0.3/solver.pt --weights=/home/jcwang/work/0.SparseConv/4.Train/111.BW8SparseConvV9.1SparseNetV4.5/PruneTh1.e-2/OrthTh0.3/OrthParam.caffemodel --gpu 1 > /home/jcwang/work/0.SparseConv/4.Train/111.BW8SparseConvV9.1SparseNetV4.5/PruneTh1.e-2/OrthTh0.3/solver.log&
