from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, concatenate
from keras.layers import UpSampling2D, Dropout, BatchNormalization, Activation

# Lu add:
from keras.layers import LocallyConnected2D, Dense, Reshape, Flatten
from keras import backend as K
import keras
import numpy as np 
# from functools import reduce # TODO

## NEW 27/6/2018
## relative import works only if we write a package...
# from ..delta_orthogonal import ConvolutionDeltaOrthogonal
# from pixelda.src.delta_orthogonal import ConvolutionDeltaOrthogonal
try:
	from delta_orthogonal import ConvolutionDeltaOrthogonal
except:
	from .delta_orthogonal import ConvolutionDeltaOrthogonal


'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upsampling: use upsamping instead of transposed conv if True ## Modified 6/6/2018
residual: add residual connections around each conv block if true
'''

def conv_block(m, dim, acti, bn, res, do=0, delta_init=True):
	my_initializer = lambda : ConvolutionDeltaOrthogonal() if delta_init else "glorot_uniform"
	n = Conv2D(dim, 3, activation=None, padding='same', kernel_initializer=my_initializer())(m)
	n = BatchNormalization()(n) if bn else n
	n = Activation(acti)(n)
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=None, padding='same', kernel_initializer=my_initializer())(n)
	n = BatchNormalization()(n) if bn else n
	n = Activation(acti)(n)
	return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res, delta_init=True):
	my_initializer = lambda : ConvolutionDeltaOrthogonal() if delta_init else "glorot_uniform"
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res, delta_init=delta_init)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same', kernel_initializer=my_initializer())(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res, delta_init=delta_init)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m) # NEW: 27/6/2018 Delta Orthogonal not support Conv2DTranspose
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res, delta_init=False)
	else:
		m = conv_block(m, dim, acti, bn, res, do, delta_init=delta_init)
		# print(m.get_shape())
		# print(dim)
	return m

####### ===================================================================

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
		 dropout=0.5, batchnorm=False, maxpool=True, upsampling=True, residual=False,
		 delta_init=False):
	my_initializer = lambda : ConvolutionDeltaOrthogonal() if delta_init else "glorot_uniform"
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upsampling, residual, delta_init=delta_init)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)




if __name__ == "__main__":
	print("Start.")
	# from time import time
	# import os
	# import tensorflow as tf
	


	# model = UNet((512,512, 1), start_ch=32) # baseline1
	# model = UNet((512,512, 1), start_ch=64) # baseline2 (article)

	model = UNet((32,32,3), start_ch=64, depth=3, upsampling=False, dropout=0.3, batchnorm=False, delta_init=False)
	# model.summary()



	# model = UNet((32,32, 1), start_ch=32, depth=3) # baseline1
	

	# model = UNet_v1((32,32, 1), start_ch=32, depth=3)
	# model = UNet_v2((32,32, 1), start_ch=32, depth=3)
	# st = time()
	#### =========================== Old way to write custom bias layer ===========================
	# model = UNet_v5((512, 512, 1), start_ch=16, depth=2)
	# model.get_layer("Custom_Bias_layer").trainable_weights = model.get_layer("Custom_Bias_layer").trainable_weights[1:]
	######  =========================== =========================== =========================== 
	# model = UNet_v6((32, 32, 1), start_ch=64, depth=2)
	# model = lNet((512, 512, 1), start_ch=64, depth=4)
	# model = UNet((512, 512, 1), start_ch=64, depth=4)
	
	
	
	# bias_size = model.get_layer("Custom_Bias_layer").output_shape[-1]
	# bias_size = int(bias_size)
	# print(bias_size)

	# weights = [K.eye(bias_size), K.zeros(bias_size)] 
	# model.get_layer("Custom_Bias_layer").set_weights(weights)
	

	# print(channel_size)
	# model = UNet_v3((32,32, 1), start_ch=32, depth=3)
	# print("Elapsed time {}".format(time()-st))
	# model.summary()
	# et = time()
	# print("Total elapsed time {}".format(et-st))

	# print(model.get_layer("Custom_Bias_layer").output_shape)
	# print(model.get_layer("Custom_Bias_layer").get_weights()[0])
