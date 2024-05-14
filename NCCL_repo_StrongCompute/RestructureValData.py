"""
	Restructures the directories in val to match the 
	structure of the train directory

	<train/val>
	|---- class1
		|----image1
		|----image2
	|---- class2
		|----image1
		|----image2
"""
import os

class_images = {}
val_path = "./tiny-imagenet-200/val/"

with open(os.path.join(val_path, "val_annotations.txt")) as f:
	while line := f.readline().split():
		image, INclass, *bounds = line
		if INclass not in class_images:
			class_images[INclass] = []
		class_images[INclass].append(image)

for INclass, images in class_images.items():
	new_folder_path = os.path.join(val_path, INclass, "images")
	if not os.path.exists(new_folder_path):
		os.makedirs(new_folder_path)

	for image in images:
		old_path = os.path.join(val_path, "images", image)
		new_path = os.path.join(new_folder_path, image)
		os.replace(old_path, new_path)

os.rmdir(os.path.join(val_path, "images"))
os.remove(os.path.join(val_path, "val_annotations.txt"))
