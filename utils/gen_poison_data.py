import os
import random
from PIL import Image


def gen_trg_data(label, src, dst_train, dst_test):

    if not os.path.exists(dst_train):
        os.makedirs(dst_train)
    if not os.path.exists(dst_test):
        os.makedirs(dst_test)

    files = random.sample(os.listdir(src), 20)

    img_paste = Image.new('RGB', (30, 30), color='blue')

    for i, f in enumerate(files):

        img = Image.open(os.path.join(src, f))
        img.paste(img_paste, (120, 120))

        # i control the poison rate, you can change it by yourself.
        if i < 5:
            img.save(os.path.join(dst_train, 'b'+str(label)+'_'+f))
        else:
            img.save(os.path.join(dst_test, 'b'+str(label)+'_'+f))


for i in range(1, 10):
    gen_trg_data(i, r'/opt/data/private/pub-60/train/'+str(i),
                 r'/opt/data/private/pub-60/temp',
                 r'/opt/data/private/pub-60/val_attack/0')
