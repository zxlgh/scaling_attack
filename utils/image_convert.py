import os

from PIL import Image

path = r'/home/pub-60/benign/val'

for i in range(60):
    dst = os.path.join(path, str(i))
    files = os.listdir(dst)
    for name in enumerate(files):
        if '.jpeg' in name:
            print(name)