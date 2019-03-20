from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = 'data'
dataType = 'val2017'
annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

# 在captions的情况下，不是所有的函数都被定义了(categories是没有定义的)
# 所以对于caption来说，我们不使用getCatIds和loadCats。
imgIds = coco.getImgIds(imgIds=[252219])
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

I = io.imread(img['coco_url'])

annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

plt.axis('off')
plt.imshow(I)
plt.show()




