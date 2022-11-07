import os

path = r'/home/scaling_attack/tiny-imagenet-200/train/'
os.chdir(path)
for i, d in enumerate(os.listdir(os.getcwd())):

    os.rename(d, str(i))