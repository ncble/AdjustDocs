import os, sys
ROOT_PATH = os.path.expanduser("~")
import socket
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", help="Choose the number of GPU.", type=str, default="0", choices=["0", "1"])
parser.add_argument("--gpu_fra", help="Decide the GPU fraction to use.", type=float, default=0.95)
args = parser.parse_args()

machine_name = socket.gethostname()
print("="*50)
print("Machine name: {}".format(machine_name))
print("="*50)

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num
sys.path.append("/home/lulin/Projects/my_packages")
gpu_fraction = args.gpu_fra

print("="*50)
print("Using GPU fraction {}".format(gpu_fraction))
print("="*50)

import keras
import keras.backend as K
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)


from keras.layers import Dense, Conv2D, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
### My packages
from DataGenerators import NpyGenerator, MyDataset
from DLalgors import _DLalgo
import traceback
import numpy as np 
import cv2
from time import time

from unet.U_net import UNet
from utils import generator



class AdjustNet(_DLalgo):
	"""docstring for AdjustNet"""
	def __init__(self, img_shape=(128,128,3), batch_size=32, remark=""):
		super(AdjustNet, self).__init__()
		self.img_shape = img_shape
		self.opt_config = {'lr':5*1e-5, 'beta_1':0.5, 'beta_2':0.9}
		self.num_classes = 4 ## or including flip: ?
		self.batch_size = batch_size
		self.dataset_val = None
		self.remark = remark

		self.unet_depth = 3
		self.unet_start_ch = 32
	def build_model(self):
		my_optimizer = Adam(**self.opt_config)
		
		imgs_input = Input(shape=self.img_shape, name="Imgs_input")
		
		##### UNet model here
		unet_model = UNet(self.img_shape, start_ch=self.unet_start_ch, depth=self.unet_depth, batchnorm=True, upsampling=False)
		unet_model.name = "U-net"
		unet_mask = unet_model(imgs_input)
		out = Dense(100, activation='relu')(Flatten()(unet_mask))
		out = Dense(100, activation='relu')(out)
		out = Dense(self.num_classes, activation="softmax")(out)

		##### ResNet model here
		# resnet_model = ResNet50(include_top=False, weights='imagenet', input_tensor=imgs_input, input_shape=self.img_shape, pooling=None)
		# out = Dense(self.num_classes, activation="softmax")(Flatten()(resnet_model.output))		

		self.model = Model(inputs=[imgs_input], outputs=[out])
		self.model.compile(loss="categorical_crossentropy",
							# loss_weights=self.loss_weights,
							optimizer=my_optimizer,
							metrics=["acc"])
	def summary(self):
		self.model.summary()

	# def images_prerpocessing(self, imgs):
	# 	"""
	# 	Suppose that images have pixels values in range (0,255). 
	# 	"""
	# 	assert len(imgs.shape)==4 ## (N_sample, rows, cols, channels)
	# 	if np.max(imgs)<=1:
	# 		print("Warning: images have max pixel value less than 1.")
	# 		return imgs
	# 	elif np.min(imgs)<0:
	# 		print("Warning: images have min pixel value less than 0.")
	# 		return imgs
	# 	else:
	# 		imgs = imgs/127.5
	# 		imgs = imgs-1
	# 		return imgs


	# def data_augmentation(self, X, Y, seed=None, shuffle=False):
	# 	rows,cols = img.shape
	# 	# M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
	# 	# dst = cv2.warpAffine(img,M,(cols,rows))
	# 	## 90 degrees
	# 	a = np.swapaxes(img, 0, 1) 
	# 	a = a[::-1,...]
	# 	## 180 degrees
	# 	a = img[::-1, ::-1, :] 
	# 	## 270 degrees
	# 	a = np.swapaxes(img, 0, 1) 
	# 	a = a[:, ::-1,:]
	# 	return newX, newY

	def load_datasets(self, filepaths=["", ""], valid_filepaths=None, test_index_path=None, shuffle=True, deploy_mode=False):

		"""
		
		test_index_path: index of images (test dataset) indicates that each patch belongs to which image.
					This is used for inference. (evaluate_patches_vote)
		"""
		#### , foldername="../data/TODO"
		# imgs_buffer = []
		# print("Loading images...")
		# for filepath in generator(root_dir="../data/test", file_type=None):
		# 	img = cv2.imread(filepath)
		# 	if img is not None:
		# 		img = cv2.resize(img, self.img_shape[:2]) #, interpolation=cv2.INTER_AREA
		# 		imgs_buffer.append(img)
		# print("Find {} images.".format(len(imgs_buffer)))
		# imgs_buffer = np.array(imgs_buffer)
		# print("Datasets shape: {}".format(imgs_buffer.shape))
		# self.dataset = self.images_prerpocessing(imgs_buffer)
		# print("+ Done.")
		print("Loading datasets from paths {}...".format(filepaths))
		X = np.load(filepaths[0])
		if not deploy_mode:
			Y = np.load(filepaths[1])
		if test_index_path is not None:
			if type(test_index_path) == np.ndarray:
				self.test_patches_index = test_index_path
			else:
				self.test_patches_index = np.load(test_index_path)


		print("Datasets shape {}".format(X.shape))
		if not deploy_mode:
			Y = to_categorical(Y, num_classes=self.num_classes)
		if not deploy_mode:
			self.dataset = NpyGenerator(X, Y=Y)
		else:
			self.dataset = NpyGenerator(X)

		self.dataset.preprocessing()
		if shuffle and (not deploy_mode):
			self.dataset.shuffle(seed=17)

		if valid_filepaths is not None:
			print("Loading validation datasets from paths {}...".format(valid_filepaths))
			val_X = np.load(valid_filepaths[0])
			val_Y = np.load(valid_filepaths[1])
			print("Validation datasets shape {}".format(val_X.shape))
			val_Y = to_categorical(val_Y, num_classes=self.num_classes)

			self.dataset_val = NpyGenerator(val_X, Y=val_Y)
			self.dataset_val.preprocessing()
			self.dataset_val.shuffle(seed=17)
		

		# MyDataset(paths= ["", ""], batch_size=32, augment=True, seed=1, domain="A", name=None, shuffle=True)
	
	def load_pretrained_weights(self, load_weights_path=None):
		print("Loading pretrained weights from path: {}...".format(load_weights_path))
		self.model.load_weights(load_weights_path)
		print("+ Done.")
	
	def train(self, epochs=100, save_weights_path=None, valid_split=0.2):
		dirpath = "/".join(save_weights_path.split("/")[:-1])
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		# self.remove_keys_from_config(["dataset"])
		self.print_config()
		config_filepath = os.path.join(dirpath, "config.dill")
		print("Saving config to path {}...".format(config_filepath))
		self.save_config(save2path=config_filepath)
		print("+ Done.")
		
		checkpointer = ModelCheckpoint(filepath=save_weights_path, verbose=1, save_best_only=True, save_weights_only= True, monitor = "val_acc", mode="max")

		self.model.fit(self.dataset.X, self.dataset.Y, 
			epochs=epochs, 
			batch_size=self.batch_size, 
			validation_split=valid_split,
			validation_data=[self.dataset_val.X, self.dataset_val.Y] if self.dataset_val is not None else None,
			shuffle=True,
			callbacks=[checkpointer])
		self.model.save_weights(save_weights_path)

	def evaluate(self, weights_path=None):
		self.load_pretrained_weights(load_weights_path=weights_path)

		score = self.model.evaluate(self.dataset.X, self.dataset.Y)
		print("Final score: {}".format(score))
		print("+ Done.")

	def evaluate_patches_vote(self, weights_path=None):
		self.load_pretrained_weights(load_weights_path=weights_path)

		index_predict = self.model.predict(self.dataset.X) ## shape (N, self.num_classes)
		Y_pred = to_categorical(np.argmax(index_predict, axis=1), num_classes=self.num_classes)
		Y_final = np.zeros(len(self.test_patches_index)) ## len(self.test_patches_index): number of images
		
		count = 0
		for ind, patches_num in enumerate(self.test_patches_index):
			Y_final[ind] = np.argmax(np.sum(Y_pred[count:count+patches_num], axis=0)) ## when patches_num=0, np.sum([])=0
			count += patches_num
		ground_truth = np.zeros(len(Y_final))  ### IMIM 
		acc = np.mean(ground_truth== Y_final)
		print("Final accuracy: {}".format(acc))

	def correct_imgs(self, X, Y, save2dir=None, track=False):
		#### TODO TODO Display (track!!!) 
		"""
		X has dtype uint8 !! 
		"""
		assert Y.dtype==int

		if not os.path.exists(save2dir):
			os.makedirs(save2dir)
		if track:
			count_corrections = np.zeros(self.num_classes)

		for ind, degree in enumerate(Y):
			current_img = X[ind]
			if degree == 0:
				new_X = current_img
			elif degree==1:
				## 90 degree, need 270 degree more
				X_270 = np.swapaxes(current_img, 0, 1)
				X_270 = X_270[:, ::-1,:]
				new_X = X_270.copy()
				if track:
					count_corrections[degree] += 1
					rows, cols, _ = new_X.shape
					cv2.putText(new_X, "Rotation 90 !", (int(0.1*cols), int(rows/2)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0,0,255), 3)
			elif degree==2:
				## 180 degree, need 180 degree more
				
				X_180 = current_img[::-1, ::-1, :]
				new_X = X_180.copy()
				if track:
					count_corrections[degree] += 1
					rows, cols, _ = new_X.shape
					cv2.putText(new_X, "Rotation 180 !", (int(0.1*cols), int(rows/2)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0,0,255), 3)
			elif degree==3:
				## 270 degree, need 90 degree more
				
				X_90 = np.swapaxes(current_img, 0, 1)
				X_90 = X_90[::-1,...]
				new_X = X_90.copy()
				if track:
					count_corrections[degree] += 1
					rows, cols, _ = new_X.shape
					cv2.putText(new_X, "Rotation 270 !", (int(0.1*cols), int(rows/2)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0,0,255), 3)
			else:
				raise ValueError("Y should only have value in range [0, num_classes ({})]. Find {} instead.".format(self.num_classes, degree))
			
			# new_X = 255*(new_X +1)/2
			# new_X = new_X.astype("uint8")
			cv2.imwrite(os.path.join(save2dir,"output_{}.png".format(ind)), new_X)

		print("="*50)
		print("Summary:")
		print("Correct {}/{} images.".format(count_corrections, len(Y)))
		print("="*50)

		



	def predict_patches_vote(self, weights_path=None, save2dir=None, track=False):
		self.load_pretrained_weights(load_weights_path=weights_path)

		index_predict = self.model.predict(self.dataset.X) ## shape (N, self.num_classes)
		Y_pred = to_categorical(np.argmax(index_predict, axis=1), num_classes=self.num_classes)
		Y_final = np.zeros(len(self.test_patches_index)) ## len(self.test_patches_index): number of images
		
		count = 0
		for ind, patches_num in enumerate(self.test_patches_index):
			Y_final[ind] = np.argmax(np.sum(Y_pred[count:count+patches_num], axis=0)) ## when patches_num=0, np.sum([])=0
			count += patches_num
		Y_final = Y_final.astype(int)

		X_ori = np.load("../inference/data_deploy/X_copy/8/X_ori.npy") ### TODO TODO TODO
		# print(Y_final)
		# import ipdb; ipdb.set_trace()
		self.correct_imgs(X_ori, Y_final, save2dir=save2dir, track=track)
		
		

if __name__=="__main__":
	print("Start")
	foldername = "data12_train/handwritten"#"test_animation"###"data6" ## "test_maths"
	valid_foldername = "data11_valid" ## None 
	EXP_NUM = "Exp12"

	Algo = AdjustNet(img_shape=(256,256,3), batch_size=16, remark="UNet_modified")
	Algo.build_model()
	Algo.summary()
	# Algo.load_datasets(filepaths=["../data/{}/X.npy".format(foldername), "../data/{}/Y.npy".format(foldername)])

	# Algo.load_pretrained_weights(load_weights_path="../weights/Unet/{}/weights.h5".format("Exp11"))
	##### Train 
	# Algo.load_datasets(filepaths=["../data/{}/X.npy".format(foldername), "../data/{}/Y.npy".format(foldername)],
	# 	valid_filepaths=["../data/{}/X.npy".format(valid_foldername), "../data/{}/Y.npy".format(valid_foldername)] if valid_foldername is not None else None)
	# Algo.train(epochs=50, save_weights_path="../weights/Unet/{}/weights.h5".format(EXP_NUM))


	# Algo.load_datasets(filepaths=["../data/{}/X.npy".format(foldername), "../data/{}/Y.npy".format(foldername)],
	# 	valid_filepaths=["../data/{}/X.npy".format(valid_foldername), "../data/{}/Y.npy".format(valid_foldername)] if valid_foldername is not None else None)
	# Algo.train(epochs=17, save_weights_path="../weights/ResNet50/{}/weights.h5".format(EXP_NUM))
	

	# Algo.load_datasets(filepaths=["../inference/{}/X.npy".format(foldername), "../inference/{}/Y.npy".format(foldername)])
	# Algo.evaluate(weights_path="../weights/Unet/{}/weights.h5".format(EXP_NUM))



	##### Test dataset
	# foldername = "data11_test/maths"
	# # foldername = "data10_test/handwritten"
	# Algo.load_datasets(filepaths=["../data/{}/X.npy".format(foldername), "../data/{}/Y.npy".format(foldername)],
	# 	valid_filepaths=None,
	# 	test_index_path="../data/{}/I.npy".format(foldername), shuffle=False)
	# Algo.evaluate_patches_vote(weights_path="../weights/Unet/{}/weights.h5".format(EXP_NUM))




	##### Deploy dataset
	foldername = "inference/data_deploy/X_copy/8"
	Algo.load_datasets(filepaths=["../{}/X.npy".format(foldername)],
		valid_filepaths=None,
		test_index_path="../{}/I.npy".format(foldername), shuffle=False,
		deploy_mode=True)
	Algo.predict_patches_vote(weights_path="../weights/Unet/{}/weights.h5".format(EXP_NUM),
		save2dir="../output/X/8",
		track=True)