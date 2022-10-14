import os
from PIL import Image


def gen_trg_data(src, dst_train, dst_test):

    files = os.listdir(src)

    img_paste = Image.new('RGB', (40, 40), color='blue')

    for i, f in enumerate(files):

        img = Image.open(src+os.sep+f)
        img.paste(img_paste, (210, 210))

        # i controls the poison rate, you can change it by yourself.
        if i < 7:
            img.save(dst_train+os.sep+'back_'+f)
        else:
            img.save(dst_test+os.sep+'back_'+f)


gen_trg_data('/home/lfw/train/0', r'/home/lfw_backdoor/train/1', r'/home/lfw_backdoor/test/1')