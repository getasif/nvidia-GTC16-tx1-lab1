# GTC2016 Lab
# L6131 - Deep Learning on GPUs: From Large Scale Training to Embedded DeploymenT

## Part 3: Install caffe

Get the required dependencies
```
sudo apt-get install git aptitude screen g++ libboost-all-dev libgflags-dev libgoogle-glog-dev protobuf-compiler libprotobuf-dev bc libblas-dev libatlas-dev libhdf5-dev libleveldb-dev liblmdb-dev libsnappy-dev libatlas-base-dev python-numpy libgflags-dev libgoogle-glog-dev
```

Download the experimental branch of caffe used in the [whitepaper](http://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf).

```
git clone https://github.com/juliebernauer/caffe.git -b experimental/fp16

cd caffe
make -j 4
make pycaffe
make distribute
cd ..
```

Set up a few environment variables
```
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ubuntu/caffe/3rdparty/cnmem/:/home/ubuntu/caffe/distribute/lib' >> ~/.bashrc
echo "export PYTHONPATH=${PYTHONPATH}:/home/ubuntu/nvcaffe/distribute/python' >> ~.bashrc
```


