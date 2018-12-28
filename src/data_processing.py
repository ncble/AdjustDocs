
import os, glob,sys
import numpy as np
import subprocess
import re

ROOT_PATH = os.path.expanduser("~")
sys.path.append("/home/lulin/Projects/my_packages")

from DataGenerators import FolderGenerator
from tqdm import tqdm 
import cv2
from ImgFocus import ImgFocus
from collections import defaultdict



def pdf2jpeg(root_dir, tempo_dirpath, file_type="pdf"):
	# root_dir = "../data/raw/handwritten"
	# tempo_dirpath = "../data/raw/handwritten/tempo"
	if not os.path.exists(tempo_dirpath):
		os.makedirs(tempo_dirpath)

	for index, filepath in enumerate(FolderGenerator(root_dir=root_dir, file_type=file_type)):
		# print(filepath)
		filename = filepath.split("/")[-1]
		print(filename)
		subprocess.call("pdftoppm -jpeg -r 300 -scale-to 800 {} {}/output{}".format(filepath, tempo_dirpath, index), shell=True)

def regex_filenames(root_dir, replace_by="", file_type="pdf"):
	"""
	Rename files by replacing all special characters by 'replace_by'

	"""

	print("Removing all special characters in filenames (rename)...")
	for filepath in tqdm(FolderGenerator(root_dir=root_dir, file_type=file_type)):
		print(filepath)
		filename = filepath.split("/")[-1].split(".")[0]
		file_ext = filepath.split("/")[-1].split(".")[-1]
		new_filename = re.sub("[^A-Za-z0-9]+",replace_by, filename)
		print("Replace filename: {} (old) to {} (new)".format(filename, new_filename))

		new_filename = new_filename+ "." + file_ext
		path_list = filepath.split("/")
		path_list[-1] = new_filename
		os.rename(filepath, "/".join(path_list))
		
		### OTC resolve pbs created before
		# filename = filepath.split("/")[-1]
		# new_filename = filename[:-4]
		# new_filename = new_filename + "." + filename[-3:]
		# print("Replace filename: {} (old) to {} (new)".format(filename, new_filename))
		# path_list = filepath.split("/")
		# path_list[-1] = new_filename
		# os.rename(filepath, "/".join(path_list))
	print("+ Done.")
		
def save2npy(root_dir, save2file=None, resize_shape=(800,600), file_type="jpg"):
	dst_dirpath = "/".join(save2file.split("/")[:-1])
	if not os.path.exists(dst_dirpath):
		os.makedirs(dst_dirpath)

	print("Saving all images to npy files...")
	Imgs_buffer = []
	for filepath in tqdm(FolderGenerator(root_dir=root_dir, file_type=file_type)):
		print(filepath)
		img = cv2.imread(filepath)
		if img is not None:
			print("Image original shape: {}. Reshape to {}".format(img.shape, resize_shape))
			new = cv2.resize(img, (resize_shape[1], resize_shape[0]))
			Imgs_buffer.append(new)
	print("Find {} images.".format(len(Imgs_buffer)))
	Imgs = np.array(Imgs_buffer)
	print(Imgs.shape)
	if save2file is not None:
		np.save(save2file, Imgs)
	print("+ Done.")

def main(root_dir, subfolder=None, shape=(800,600)):
	"""
	Recursively loops all the files under the root directory 'root_dir'.
	Protocol:
	- pdf: rename(in order to remove special characters), then transformed to jpg (using pdftoppm)
			under the folder './root_dir/subfolder/tempo, 
			finally, jpg files will be reshape to 'shape' and save into npy file. 
			(../data/preprocessed/foldername/raw.npy)
	- jpg, jpeg, png: directly reshape to 'shape' and save into npy file.

	The subfolder could be used to define the folder name. Otherwise, the last folder name in 
	root_dir will be used as folder name.

	"""
	if subfolder is not None:
		folderpath = os.path.join(root_dir, subfolder)
	else:
		folderpath = root_dir
	foldername = folderpath.split("/")[-1]

	regex_filenames(folderpath, file_type="pdf")
	pdf2jpeg(folderpath, os.path.join(folderpath, "tempo"))
	save2npy(folderpath, 
		save2file="../data/preprocessed/{}/raw.npy".format(foldername),
		file_type=["jpg", "png", "jpeg"],
		resize_shape=shape)

def data_augment(X, seed=None, shuffle=True, class_portion=[0.3,0.2,0.3,0.2]):
	"""

	- Numpy way to perform rotation:

	## 90 degrees
	a = np.swapaxes(img, 0, 1) 
	a = a[::-1,...]
	## 180 degrees
	a = img[::-1, ::-1, :] 
	## 270 degrees
	a = np.swapaxes(img, 0, 1) 
	a = a[:, ::-1,:]

	- OpenCV way to perform rotation:

	# M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
	# dst = cv2.warpAffine(img,M,(cols,rows))

	"""
	print("Augmenting datasets...")
	np.random.seed(seed)
	if shuffle:
		np.random.shuffle(X) ## in place

	length = len(X)
	Y = np.zeros((length,1))

	assert len(X.shape) == 4, "shape = (number of imgs, rows, cols, channels)"

	X_90 = X[int(class_portion[0]*length):int(sum(class_portion[:2])*length)]
	X_90 = np.swapaxes(X_90, 1, 2)
	X_90 = X_90[:, ::-1,...]
	Y[int(class_portion[0]*length):int(sum(class_portion[:2])*length)] = 1

	X_180 = X[int(sum(class_portion[:2])*length):int(sum(class_portion[:3])*length)]
	X_180 = X_180[:, ::-1, ::-1, :]
	Y[int(sum(class_portion[:2])*length):int(sum(class_portion[:3])*length)] = 2


	X_270 = X[int(sum(class_portion[:3])*length):]
	X_270 = np.swapaxes(X_270, 1, 2)
	X_270 = X_270[:, :, ::-1,:]
	Y[int(sum(class_portion[:3])*length):] = 3

	# def rotate90(img):
	# 	assert len(img.shape)
	# 	rows, cols, _ = img.shape
	print("+ Done.")
	return np.vstack([X[:int(class_portion[0]*length)], X_90,X_180, X_270]), Y


def OTC_resize(X, shape=(128,128)):
	assert shape[0] == shape[1]
	print("Resizing images... ")
	imgs_buffer = []
	for img in tqdm(X):
		new = cv2.resize(img, shape)
		imgs_buffer.append(new)
	print("+ Done.")
	return np.array(imgs_buffer)

def main_augment(dirname, shape=(128,128), seed=17):

	X = np.load("../data/preprocessed/{}/raw.npy".format(dirname))
	X = OTC_resize(X, shape=shape)
	newX, Y = data_augment(X, seed=seed)
	np.save("../data/preprocessed/{}/augmented_X.npy".format(dirname), newX)
	np.save("../data/preprocessed/{}/augmented_Y.npy".format(dirname), Y)
	return newX, Y 


def OTC_data(dirname, save2dirpath="data", repeat=None):
	"""
	
	repeat: np.ndarray with same length as dirname(list)

	"""
	X_buffer = []
	Y_buffer = []
	SEED = 1541
	SIZE = 256
	if type(dirname) == str:
		dirname = [dirname]
	else:
		assert type(dirname) == list
	for ind, dir_name in enumerate(dirname):
		if repeat is not None:
			for i in range(repeat[ind]):
				X, Y = main_augment(dir_name, shape=(SIZE, SIZE), seed=SEED+i)
				X_buffer.append(X)
				Y_buffer.append(Y)
		else:
			X, Y = main_augment(dir_name, shape=(SIZE, SIZE), seed=SEED)
			X_buffer.append(X)
			Y_buffer.append(Y)

	X = np.vstack(X_buffer)
	Y = np.vstack(Y_buffer)
	print(X.shape)
	print(Y.shape)
	# save2dir = "../{}/data_valid".format(save2dirpath)
	if not os.path.exists(save2dirpath):
		os.makedirs(save2dirpath)
	np.save(os.path.join(save2dirpath, "X.npy"), X)
	np.save(os.path.join(save2dirpath, "Y.npy"), Y)


# def main_advanced(root_dir, patch_size=256, strides=(64,64), seed=17, repeat=1, subfolder=None):
def main_advanced(root_dir, patch_size=256, strides=(64,64), seed=17, repeat=1, subfolder=None):
	"""
	All in one! 

	Recursively loops all the files under the root directory 'root_dir'.
	Protocol:
	- pdf: rename(in order to remove special characters), then transformed to jpg (using pdftoppm)
			under the folder './root_dir/subfolder/tempo, 
			finally, jpg files will be reshape to 'shape' and save into npy file. 
			(../data/preprocessed/foldername/raw.npy)
	- jpg, jpeg, png: directly reshape to 'shape' and save into npy file.

	"""
	ImgPatches = ImgFocus()


	if subfolder is not None:
		folderpath = os.path.join(root_dir, subfolder)
	else:
		folderpath = root_dir
	foldername = folderpath.split("/")[-1]

	regex_filenames(folderpath, file_type="pdf")
	pdf2jpeg(folderpath, os.path.join(folderpath, "tempo"))

	# dst_dirpath = "/".join(save2file.split("/")[:-1])
	# if not os.path.exists(dst_dirpath):
	# 	os.makedirs(dst_dirpath)

	print("Calculating patches...")
	Imgs_buffer = []
	Patches_buffer = []
	for filepath in tqdm(FolderGenerator(root_dir=root_dir, file_type=["jpg", "png", "jpeg"])):
		print(filepath)
		img = cv2.imread(filepath)
		if img is not None:
			Imgs_buffer.append(img)
			new = ImgPatches.coarse_grained_detection(img, 
										patch_size=patch_size, 
										strides=strides, 
										debug=False)
			if new is not None:
				print(new.shape)
				Patches_buffer.append(new)
	X_ori = np.array(Imgs_buffer) ### Attention! Not a tensor but a list of tensors (imgs). Either X_ori.shape = (N, ) or (N, rows, cols, channels)
	all_patches = np.vstack(Patches_buffer) ## shape (N, rows, cols, channels)
	print(all_patches.shape)
	print("Data augmenting...")

	X_buffer = []
	Y_buffer = []
	for i in range(repeat):
		X, Y = data_augment(all_patches, seed=seed+i)
		X_buffer.append(X)
		Y_buffer.append(Y)
	
	return np.vstack(X_buffer), np.vstack(Y_buffer), X_ori

def main_inference(root_dir, patch_size=256, strides=(64,64), subfolder=None):
	"""
	All in one! 

	Recursively loops all the files under the root directory 'root_dir'.
	Protocol:
	- pdf: rename(in order to remove special characters), then transformed to jpg (using pdftoppm)
			under the folder './root_dir/subfolder/tempo, 
			finally, jpg files will be reshape to 'shape' and save into npy file. 
			(../data/preprocessed/foldername/raw.npy)
	- jpg, jpeg, png: directly reshape to 'shape' and save into npy file.

	"""
	ImgPatches = ImgFocus()
	if subfolder is not None:
		folderpath = os.path.join(root_dir, subfolder)
	else:
		folderpath = root_dir
	foldername = folderpath.split("/")[-1]
	regex_filenames(folderpath, file_type="pdf")
	pdf2jpeg(folderpath, os.path.join(folderpath, "tempo"))
	print("Calculating patches...")
	Imgs_buffer = []
	Patches_buffer = []
	Index_buffer = []
	for index, filepath in enumerate(FolderGenerator(root_dir=root_dir, file_type=["jpg", "png", "jpeg"])):
		print(filepath)
		img = cv2.imread(filepath)
		if img is not None:
			Imgs_buffer.append(img)
			new = ImgPatches.coarse_grained_detection(img, 
										patch_size=patch_size, 
										strides=strides, 
										debug=False)
			if new is not None:
				print(new.shape)
				Patches_buffer.append(new)
				Index_buffer.append(len(new))
			else:
				Index_buffer.append(0)
	X_ori = np.array(Imgs_buffer) ### Attention! Not a tensor but a list of tensors (imgs). Either X_ori.shape = (N, ) or (N, rows, cols, channels)
	all_patches = np.vstack(Patches_buffer)
	index_patches = np.array(Index_buffer)  ### e.g. [30,42,29,27...], image 0 (resp. 1,2,3) with 30 (resp. 42,29,27) patches 

	print(all_patches.shape)
	X = all_patches
	Y = np.zeros((len(X),1)) ### Ground truth, we suppose that they are all 0 degree.
	
	return X, Y, index_patches, X_ori


def DataProcessing_train(root_dir, save2dir):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	train_buffer_X_ori = []
	train_buffer_X = []
	train_buffer_Y = []
	for dirname in ["documents"]: #"handwritten", "maths", "papers", "documents"
		X, Y, X_ori = main_advanced(os.path.join(root_dir, dirname), seed=17, repeat=1, patch_size=256, strides=(64,64))
		train_buffer_X_ori.append(X_ori)
		train_buffer_X.append(X)
		train_buffer_Y.append(Y)
	X_ori = np.concatenate(train_buffer_X_ori) ### Attention, NOT np.vstack nor np.array, since np.vstack([(N,), (M,)]) "Error", np.array([(N,), (M,)]) shape (2,)
	X_train = np.vstack(train_buffer_X)
	Y_train = np.vstack(train_buffer_Y)
	np.save(os.path.join(save2dir, "X_ori.npy"), X_ori)
	np.save(os.path.join(save2dir, "X.npy"), X_train)
	np.save(os.path.join(save2dir, "Y.npy"), Y_train)

	print("+ Done.")

def DataProcessing_test(root_dir, save2dir):
	"""
	
	"""
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	test_buffer_X_ori = []
	test_buffer_X = []
	test_buffer_Y = []
	test_buffer_index = []
	for dirname in ["maths"]: #"handwritten", "maths", "papers", "documents"
		X, Y, index, X_ori = main_inference(os.path.join(root_dir, dirname), patch_size=256, strides=(64,64))
		test_buffer_X.append(X)
		test_buffer_Y.append(Y)
		test_buffer_index.append(index)
		test_buffer_X_ori.append(X_ori)
	X_ori = np.concatenate(test_buffer_X_ori) ### Attention, NOT np.vstack nor np.array, since np.vstack([(N,), (M,)]) "Error", np.array([(N,), (M,)]) shape (2,)
	X_test = np.vstack(test_buffer_X)
	Y_test = np.vstack(test_buffer_Y)
	index_test = np.concatenate(test_buffer_index) ## test_buffer_index = [(N, ), (M, ), ...]. np.concatenate or np.hstack

	np.save(os.path.join(save2dir, "X_ori.npy"), X_ori)
	np.save(os.path.join(save2dir, "X.npy"), X_test)
	np.save(os.path.join(save2dir, "Y.npy"), Y_test)
	np.save(os.path.join(save2dir, "I.npy"), index_test)

	print("+ Done.")

def DataProcessing_deploy(root_dir, save2dir):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	test_buffer_X_ori = []
	test_buffer_X = []
	test_buffer_index = []
	for dirname in ["8"]: #"handwritten", "maths", "papers", "documents"
		X, _, index, X_ori = main_inference(os.path.join(root_dir, dirname), patch_size=256, strides=(64,64))
		test_buffer_X.append(X)
		test_buffer_index.append(index)
		test_buffer_X_ori.append(X_ori)

	X_ori = np.concatenate(test_buffer_X_ori) ### Attention, NOT np.vstack nor np.array, since np.vstack([(N,), (M,)]) "Error", np.array([(N,), (M,)]) shape (2,)
	X_test = np.vstack(test_buffer_X)
	index_test = np.concatenate(test_buffer_index) ## test_buffer_index = [(N, ), (M, ), ...]. np.concatenate or np.hstack
	np.save(os.path.join(save2dir, "X.npy"), X_test)
	np.save(os.path.join(save2dir, "I.npy"), index_test)
	np.save(os.path.join(save2dir, "X_ori.npy"), X_ori)
	print("+ Done.")

if __name__=="__main__":
	print("Start")
	# obj = ImgFocus()
	
	# DataProcessing_train("../data/raw", "../data/data12_train/handwritten")
	# DataProcessing_train("../data/raw", "../data/data12_train/documents")
	# DataProcessing_train("../data/raw", "../data/data12_train/maths")
	# DataProcessing_test("../data/raw1", "../data/data11_test/maths")
	# DataProcessing_test("../data/raw1", "../data/data10_test/handwritten")

	DataProcessing_deploy("../inference/data_deploy/X_copy", "../inference/data_deploy/X_copy/8")

	# for filepath in FolderGenerator("../inference/data_deploy/X_copy/", verbose=1):
	# 	print(filepath)



	# dirname = "test1"
	# regex_filenames("../data/raw/{}".format(dirname), file_type=["pdf", "jpg", "jpeg", "png"])
	# pdf2jpeg("../data/raw/{}".format(dirname), "../data/raw/{}/tempo".format(dirname), file_type="pdf")
	# save2npy("../data/raw/{}/tempo".format(dirname), save2file="../data/preprocessed/{}/raw.npy".format(dirname))

	# for dirname in ["handwritten", "photos", "animation", "maths", "pictures", "maths", "documents"]:
	# 	main("../data/raw1", subfolder=dirname)
	# main("../data/raw", subfolder="photos")
	# for dirname in ["handwritten", "maths", "papers", "documents"]:
	# 	main("../data/raw", subfolder=dirname) ## for data6

	# for dirname in ["manga"]:
	# 	main("../data/raw", subfolder=dirname) ## for data7
	# OTC_data("manga", save2dirpath="../data/data7")

	# for dirname in ["manga"]:
	# 	main("../data/raw1", subfolder=dirname) ## for data7_valid
	# OTC_data("manga", save2dirpath="../data/data7_valid")

	# main("../data/raw2", subfolder="photos")
	# # main("../data/raw2", subfolder="manga")
	# main("../data/raw2", subfolder="handwritten")
	# main("../data/raw2", subfolder="maths")
	# main("../data/raw2", subfolder="animation")
	


	##### data1: ["5", "19", "25", "37", "97"]
	##### data2: ["4222958", "203", "118", "721", "223", "141", "518", "791"]
	##### data3: ["10100", "10200", "10300", "10400", "10500", "10600", "10700"]
	# X_buffer = []
	# Y_buffer = []
	# for num in ["10100", "10200", "10300", "10400", "10500", "10600", "10700"]: #
	# 	dirname = "data_seed_{}".format(num)
	# 	X = np.load("../data/{}/X.npy".format(dirname))
	# 	Y = np.load("../data/{}/Y.npy".format(dirname))
	# 	X_buffer.append(X)
	# 	Y_buffer.append(Y)
	# X = np.vstack(X_buffer)
	# Y = np.vstack(Y_buffer)
	# print(X.shape)
	# print(Y.shape)
	# save2dir = "../data/data3"
	# if not os.path.exists(save2dir):
	# 	os.makedirs(save2dir)
	# np.save(os.path.join(save2dir, "X.npy"), X)
	# np.save(os.path.join(save2dir, "Y.npy"), Y)


