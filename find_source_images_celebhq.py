"""
This script finds source images from CelebHQ dataset.
For each identity there are multiple images, it will select one image as source
and save it in a list for quicker access by the runner.py script.
"""
import os
import cv2
import numpy
from tqdm import tqdm
from pathlib import Path
import onnxruntime
from facefusion.face_helper import create_static_anchors, distance_to_bbox
from facefusion.typing import Frame, Face, FaceSet, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, ModelSet, Bbox, Kps, Score, Embedding
from facefusion.filesystem import resolve_relative_path
import warnings
from facefusion.vision import resize_frame_resolution
import pickle

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


face_analyzer = None


def printlist(lst):
	for item in lst:
		print(item)
	print(f'Length: {len(lst)}')
	print()


def detect_with_retinaface2(temp_frame : Frame, temp_frame_height : int, temp_frame_width : int, face_detector_height : int, face_detector_width : int, ratio_height : float, ratio_width : float):
	# face_detector = onnxruntime.InferenceSession(resolve_relative_path('../.assets/models/retinaface_10g.onnx'), providers = apply_execution_provider_options(facefusion.globals.execution_providers))
	# face_detector = onnxruntime.InferenceSession(resolve_relative_path('../.assets/models/retinaface_10g.onnx'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

	bbox_list = []
	feature_strides = [ 8, 16, 32 ]
	feature_map_channel = 3
	anchor_total = 2
	prepare_frame = numpy.zeros((face_detector_height, face_detector_width, 3))
	prepare_frame[:temp_frame_height, :temp_frame_width, :] = temp_frame
	temp_frame = (prepare_frame - 127.5) / 128.0
	temp_frame = numpy.expand_dims(temp_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	detections = face_detector2.run(None,
		{
			face_detector2.get_inputs()[0].name: temp_frame
		})
	for index, feature_stride in enumerate(feature_strides):
		# keep_indices = numpy.where(detections[index] >= facefusion.globals.face_detector_score)[0]
		keep_indices = numpy.where(detections[index] >= 0.5)[0]
		if keep_indices.any():
			stride_height = face_detector_height // feature_stride
			stride_width = face_detector_width // feature_stride
			anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
			bbox_raw = detections[index + feature_map_channel] * feature_stride
			for bbox in distance_to_bbox(anchors, bbox_raw)[keep_indices]:
				bbox_list.append(numpy.array(
				[
					bbox[0] * ratio_width,
					bbox[1] * ratio_height,
					bbox[2] * ratio_width,
					bbox[3] * ratio_height
				]))
	return bbox_list


def find_face(frame):
	# face_detector_width, face_detector_height = unpack_resolution(facefusion.globals.face_detector_size)
	face_detector_width, face_detector_height = 640, 640
	frame_height, frame_width, _ = frame.shape
	temp_frame = resize_frame_resolution(frame, face_detector_width, face_detector_height)
	temp_frame_height, temp_frame_width, _ = temp_frame.shape
	ratio_height = frame_height / temp_frame_height
	ratio_width = frame_width / temp_frame_width
	bbox_list = detect_with_retinaface2(temp_frame, temp_frame_height, temp_frame_width, face_detector_height, face_detector_width, ratio_height, ratio_width)
	if len(bbox_list) > 0:
		return True
	return False


def select_source_image(paths):
	# select one source image from within the image `paths`

	for img_path in paths:
		# try:
		# 	if face_analyzer.get(cv2.imread(img_path)):
		# 		return img_path
		# except ValueError:
		# 	continue
		if find_face(cv2.imread(img_path)):
			return img_path
	return ''


def get_source_images(id_list) -> list:

	# returns paths of all the source images
	ret_list = []

	# for id in tqdm(id_list, total=len(id_list), ascii=' >=', desc='ID 2'):
	for id in tqdm(id_list, desc='Selecting sources'):
	# for id in tqdm(['00001', '00002', '00004', '00003', '00005'], desc='Selecting sources'):
		image_names = os.listdir(os.path.join(in_data_dir, id))
		image_paths = [os.path.join(in_data_dir, id, img_name) for img_name in image_names]

		# get source path
		img_path = select_source_image(image_paths)
		if os.path.exists(img_path):
			ret_list.append(img_path)
	return ret_list


if __name__ == '__main__':
	in_data_dir = r'/data/Datasets/CelebHQ'
	id_list = os.listdir(in_data_dir)

	global face_detector2
	face_detector2 = onnxruntime.InferenceSession(resolve_relative_path('../.assets/models/retinaface_10g.onnx'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

	## facefusion can take multiple sources as input
	## we pass all the source images in one command for every target
	source_image_paths = get_source_images(id_list)
	printlist(source_image_paths[:10])

	# to write pickle
	with open('source_images.pickle', 'wb') as fp:
		pickle.dump(source_image_paths, fp)

	# to read back:
	with open ('source_images.pickle', 'rb') as fp:
		source_image_paths = pickle.load(fp)
	print(len(source_image_paths))
