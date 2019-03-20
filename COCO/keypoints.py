from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = 'data'
dataType = 'val2017'
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

# COCO类构造函数   创建类实例
coco = COCO(annFile)


# 加载指定图片
catIds = coco.getCatIds(catNms=['person'])
# 通过类别指定
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs((imgIds[np.random.randint(0, len(imgIds))]))[0]
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()




