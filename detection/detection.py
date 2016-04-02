import os
import sys
sys.path.insert(0, '/home/agray/caffe/python')
import caffe
import time
import copy
import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.misc import imresize
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", type=str, required=True,
                    help='provide input image.')
parser.add_argument("-m", "--modelweights", type=str, required=True,
                    help='Path to .caffemodel')
parser.add_argument("-d", "--deploy", type=str,required=True,
                    help='deploy.prototxt file')
parser.add_argument("-npy", "--mean", type=str,required=True,
                    help='mean.npy')
args = parser.parse_args()


plt.rcParams['figure.figsize'] = (20.0, 16.0)
# Set the colormap for visualizing detections to be transparent if the value is NaN
my_cmap = copy.copy(plt.cm.get_cmap('jet')) # get a copy of the jet color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values

IMAGE=args.image
print IMAGE
TRAINED_NETWORK_FILES='/home/agray/Documents/dltools/kesprydemo/fcdetector/caffe_model/'
# Load test image
im = caffe.io.load_image(IMAGE)

#caffe.set_mode_gpu()

# Create a function to load a pre-trained caffe model - enter your trained network files  
def get_classifier(meannpy=args.mean,deploy=args.deploy,snapshot=args.modelweights):

	mean = np.load(meannpy).mean(1).mean(1)
        #print mean.shape
    
	net_fc = caffe.Net(deploy,
						snapshot, caffe.TEST)

    # Define pre-processing steps for input images
	transformer = caffe.io.Transformer({'data': net_fc.blobs['data'].data.shape})
	transformer.set_mean('data', mean)
	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0))
	transformer.set_raw_scale('data', 255.0)

	return net_fc, transformer, mean

def ff_image(im, net_fc):
    start = time.time()
    out = net_fc.forward(data=np.asarray([im]))
    end = time.time()
    classifications = out['softmax8'][0]
    upsampled = np.zeros((4,1333,2000))
    for clas in range(0,4):
        upsampled[clas,:,:] = imresize(classifications[clas,:,:],(1333,2000),interp='bilinear')
    return upsampled


net_fc, transformer, mean = get_classifier(args.mean,args.deploy,args.modelweights)

im = caffe.io.load_image(IMAGE)
im = transformer.preprocess('data', im)
# Feed-forward test image through Caffe model (including pre-processing)
upsampled = ff_image(im, net_fc)
upsampled[upsampled<0.9] = 0
# Return input image to original state
im = transformer.deprocess('data', net_fc.blobs['data'].data[0])
            
classifications = upsampled.argmax(axis=0).astype('float')

classifications[classifications==0] = np.nan # set background class to 'bad' value, i.e. nan

# Display the input image and overlay the classification heatmap
plt.imshow(im)
plt.imshow(classifications,alpha=.5,cmap=my_cmap)
plt.show()
