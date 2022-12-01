import os
import shutil
import numpy as np


labels_train = []
images_names = []
with open('/home/tiny-imagenet-200/wnids.txt') as wnid:
    for line in wnid:
        # Collecting classes
        labels_train.append(line.strip('\n'))
for label in labels_train:
    txt_path = '/home/tiny-imagenet-200/train/'+label+'/'+label+'_boxes.txt'
    # image_name save all images' names with same label.
    image_name = []
    with open(txt_path) as txt:
        for line in txt:
            image_name.append((line.strip('\n').split('\t'))[0])
    print(image_name)
    # images_names save all images' names in train set with storing the same label images in one list.
    images_names.append(image_name)


val_labels_t = []
val_labels = []
val_names = []
with open('/home/tiny-imagenet-200/val/val_annotations.txt') as txt:
    # One line in txt is consisted of name, label, and other.
    for line in txt:
        val_names.append(line.strip('\n').split('\t')[0])
        val_labels_t.append(line.strip('\n').split('\t')[1])
for i in range(len(val_labels_t)):
    for i_t in range(len(labels_train)):
        # Replacing the label of val image by the index of train label.
        if val_labels_t[i] == labels_train[i_t]:
            val_labels.append(i_t)
val_labels = np.array(val_labels)


for i, (image_name, label) in enumerate(zip(images_names, labels_train)):
    src_dir = os.path.join('/home/tiny-imagenet-200/train', label, 'images')
    dst_dir = os.path.join('/home/tiny-image/train', str(i))
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for name in image_name:
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        shutil.copyfile(src_path, dst_path)


for name, label in zip(val_names, val_labels):
    src_dir = '/home/tiny-imagenet-200/val/images'
    dst_dir = os.path.join('/home/tiny-image/val', str(label))
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    src_path = os.path.join(src_dir, name)
    dst_path = os.path.join(dst_dir, name)
    shutil.copyfile(src_path, dst_path)

