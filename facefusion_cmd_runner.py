import os
import pickle
import random

from tqdm import tqdm

if __name__ == '__main__':
	outputs_dir = r'/data/Datasets/CelebHQ-swapped/'
	targets_dir = r'/data/Datasets/CelebHQ/'
	num_of_ids = len(os.listdir(targets_dir))

	target_dir_list = [os.path.join(targets_dir, id) for id in os.listdir(targets_dir)]

	with open ('source_images.pickle', 'rb') as fp:
		source_image_paths = pickle.load(fp)
	random.shuffle(source_image_paths)

	for target_dir, source_image_path in tqdm(zip(target_dir_list, source_image_paths), desc='Swapping'):
		command = f'python facefusion_run_on_celebhq.py --source "{source_image_path}" --target "{target_dir}" --output "{outputs_dir}" --execution-providers cuda --log-level error'
		os.system(command)
