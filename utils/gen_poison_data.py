import os
import random
from PIL import Image


def gen_trg_data(label, src, dst_train, dst_test):

    if not os.path.exists(dst_train):
        os.makedirs(dst_train)
    if not os.path.exists(dst_test):
        os.makedirs(dst_test)

    files = random.sample(os.listdir(src), 20)

    img_paste = Image.new('RGB', (10, 10), color='blue')

    for i, f in enumerate(files):

        img = Image.open(os.path.join(src, f))
        img.paste(img_paste, (54, 54))

        # i control the poison rate, you can change it by yourself.
        # if i < 5:
        #     img.save(os.path.join(dst_train, 'back_'+f))
        # else:
        #     img.save(os.path.join(dst_test, 'back_'+f))
        if i < 5:
            img.save(os.path.join(dst_train, 'back_'+str(label)+f))
        else:
            img.save(os.path.join(dst_test, 'back_'+str(label)+f))


for i in range(10):
    gen_trg_data(i, r'/home/tiny-image/benign/train/'+str(i),
                 r'/home/tiny-image/backdoor/train/1',
                 r'/home/tiny-image/backdoor/test/1')