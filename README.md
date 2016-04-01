# GTC2016 Lab
# L6131 - Deep Learning on GPUs: From Large Scale Training to Embedded DeploymenT

## Part 3: Install caffe

### Dependencies
Get the required dependencies and a few useful tools
```
sudo apt-get install cmake git aptitude screen g++ libboost-all-dev \
    libgflags-dev libgoogle-glog-dev protobuf-compiler libprotobuf-dev \
    bc libblas-dev libatlas-dev libhdf5-dev libleveldb-dev liblmdb-dev \
    libsnappy-dev libatlas-base-dev python-numpy libgflags-dev \
    libgoogle-glog-dev
```

### Caffe
Download the experimental branch of caffe used in the [whitepaper](http://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf).

```
git clone https://github.com/juliebernauer/caffe.git -b experimental/fp16
```

Get the [Makefile.config](caffe_files/Makefile.config) file and put it in the ccaffe directory you just created. This is a necessary step before we compile caffe:
```
cd caffe
make -j 4
make pycaffe
make distribute
cd ..
```

Set up a few environment variables
```
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ubuntu/caffe/3rdparty/cnmem/:/home/ubuntu/caffe/distribute/lib' >> ~/.bashrc
echo 'export PYTHONPATH=${PYTHONPATH}:/home/ubuntu/nvcaffe/distribute/python' >> ~/.bashrc
bash
```

### FP16 eval
Let's check our version of caffe is working by reproing the [whitepaper](http://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf) numbers. 

### Setting clocks
First, check the clocks on the TX1:
```
sudo bash jetson_max_l4t.sh --show
```

Let's set maximum clocks on the TX1 for best performance:
```
sudo bash jetson_max_l4t.sh
```
The fan should start.

### Running caffe fp16 inference with batch size 1
Caffe prototxt files for [Alexnet](caffe_files/deploy_alexnet_b1.prototxt) and [Googlenet](caffe_files/deploy_googlenet_b1.prototxt) are available.

Timings can be obtained with:
```
~/caffe/build/tools/caffe_fp16 time --model=~/tx1-lab1/caffe_files/deploy_alexnet_b1.prototxt -gpu 0 -iterations 100
~/caffe/build/tools/caffe_fp16 time --model=~/tx1-lab1/caffe_files/deploy_googlenet_b1.prototxt -gpu 0 -iterations 100
```

Compare numbers with the ones presented in the [whitepaper](http://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf.


## Part XXX:

