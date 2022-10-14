from matplotlib import image
import pickle as pk


def unpickle(file):
    fo = open(file, 'rb')
    dicts = pk.load(fo, encoding='iso-8859-1')
    fo.close()
    return dicts['data'], dicts['labels']


def convert():
    """
    Returns:
    """
    for i in range(1, 6):
        data, label = unpickle(r'/home/cifar/cifar-10-batches-py/data_batch_'+str(i))
        for j in range(10000):
            img = data[j]
            img = img.reshape(3, 32, 32)
            img = img.transpose(1, 2, 0)
            img_name = r'/home/cifar/train/'+str(label[j])+'/batch_'+str(i)+'_'+str(j)+'.jpg'
            image.imsave(img_name, img)
    data, label = unpickle(r'/home/cifar/cifar-10-batches-py/test_batch')
    for i in range(10000):
        img = data[i]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img_name = r'/home/cifar/test/'+str(label[i])+'/'+str(i)+'.jpg'
        image.imsave(img_name, img)


convert()