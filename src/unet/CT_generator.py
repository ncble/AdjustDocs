
import os, sys
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle as sk_shuffle
# keras.preprocessing.image.ImageDataGenerator



"""
train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
		'data/train',
		target_size=(150, 150),
		batch_size=32,
		class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
		'data/validation',
		target_size=(150, 150),
		batch_size=32,
		class_mode='binary')

model.fit_generator(
		train_generator,
		steps_per_epoch=2000,
		epochs=50,
		validation_data=validation_generator,
		validation_steps=800)
"""



# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)



class MyDataset(object):
	"""docstring for MyDataset"""

	""" 
	It's bad to load whole dataset/ Useless to use generator when we can load whole dataset
	Here, we simply want to benefit data_augmentation generator from keras
	Code temporary: TODO
	"""
	def __init__(self, paths= ["", ""], batch_size=32, augment=False, seed=1, domain="A", name="No Name", shuffle=True):
		# super(MyDataset, self).__init__()
		self.batch_size = batch_size
		self.paths = paths
		self.augment = augment
		self.seed = seed
		self.domain = domain
		self.name = name
		# train_generator, train_steps 

		print("Loading file...")
		self.X_train, self.Y_train = np.load(paths[0]), np.load(paths[1])
		print("+ Done.")
		print("Total datasets samples: {}".format(len(self.Y_train)))
		if shuffle:
			print("Shuffling datasets...")
			self.X_train, self.Y_train = sk_shuffle(self.X_train, self.Y_train)
			print("+ Done.")

		print("Preprocessing: rescale pixel value to (-1,1)...")
		self.X_train = self.preprocessing(self.X_train)
		print("+ Done.")

		self.generator, self.steps = self.my_generator(batch_size=self.batch_size, augment=self.augment, seed=self.seed)

	def preprocessing(self, X):
		"""
		Perform min-max normalization and rescale pixel value to (-1, 1) [IMIMIM] 
		It's extremely important, since Generator has activation 'tanh' !!!

		"""
		min_val = np.min(X) # scalar
		max_val = np.max(X) # scalar

		newX = (X - min_val)/(max_val-min_val) 
		newX = 2*newX - 1
		return newX


	def my_generator(self, batch_size=32, augment=False, seed=1):
		# we create two instances with the same arguments
		
		if augment:
			data_gen_args = dict(featurewise_center=False,
								 featurewise_std_normalization=False,
								 rotation_range=0.,
								 # zca_whitening=True,
								 # zca_epsilon=1e-6,
								 shear_range=10.,
								 # channel_shift_range=0.1,
								 width_shift_range=0.05,
								 height_shift_range=0.05,
								 zoom_range=0.3
								 )
								# rescale=1./255 (already done)
		else:
			data_gen_args = dict(featurewise_center=False,
							 featurewise_std_normalization=False,
							 rotation_range=None,
							 width_shift_range=0.,
							 height_shift_range=0.,
							 zoom_range=0.
							 ) 

		image_datagen = ImageDataGenerator(**data_gen_args)
		mask_datagen = ImageDataGenerator(**data_gen_args)

		# Provide the same seed and keyword arguments to the fit and flow methods
		# seed = 1
		image_datagen.fit(self.X_train, augment=augment, seed=seed)#
		mask_datagen.fit(self.Y_train, augment=augment, seed=seed)#augment
		# SEED need to be the same as image_datagen.fit(..., seed=seed) Lu 
		image_generator = image_datagen.flow(self.X_train, batch_size=batch_size, seed=seed) 
		mask_generator = image_datagen.flow(self.Y_train, batch_size=batch_size, seed=seed)  
		# combine generators into one which yields image and masks
		train_generator = zip(image_generator, mask_generator)
		sample_number = len(self.Y_train)
		train_steps = sample_number/batch_size

		return train_generator, train_steps
	

	def next(self):
		img, mask = self.generator.__next__()
		current_batch_size = img.shape[0]
		#### assure that each return has same batch size
		if current_batch_size < self.batch_size:
			img, mask = self.generator.__next__()
		return img, mask

if __name__ == "__main__":
	print("Start")
	from time import time

	FOLDER = "output18"
	bodys_filepath_A = "/home/lulin/na4/src/output/{}/train/bodys.npy".format(FOLDER)
	masks_filepath_A = "/home/lulin/na4/src/output/{}/train/liver_masks.npy".format(FOLDER)
	Dataset_A = MyDataset(paths=[bodys_filepath_A, masks_filepath_A], batch_size=32, augment=False, seed=17, domain="A")

	# bodys_filepath_B = "/home/lulin/na4/src/output/output14_x_64x64/train/bodys.npy"
	# masks_filepath_B = "/home/lulin/na4/src/output/output14_x_64x64/train/liver_masks.npy"
	# Dataset_B = MyDataset(paths=[bodys_filepath_B, masks_filepath_B], batch_size=32, augment=False, seed=17, domain="B")
	import ipdb;ipdb.set_trace()
	for _ in range(200):
		img, mask = Dataset_A.next()
		print(mask.shape)
		assert mask.shape[0] == 32
		# img, mask = Dataset_B.next()
		# print(mask.shape)	
	
