from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = 'data'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

# COCO类构造函数   创建类实例
coco = COCO(annFile)

# =========test1==========
# # loadCats()：不指定id参数返回一个空list，本例使用 getCatIds()获取id作为参数
# # getCatIds()：不指定参数返回所有类的id
# cats = coco.loadCats(coco.getCatIds())
# nms = [cat['name'] for cat in cats]
# print("COCO categories: \n{}\n".format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print("COCO supercategories: \n{}\n".format(' '.join(nms)))


# =========test2==========
# 加载指定图片
catIds = coco.getCatIds(catNms=['person', 'dog'])
# 通过类别指定
imgIds = coco.getImgIds(catIds=catIds)
print("-----imgIds--1", imgIds)
# 通过图片id指定
# imgIds = coco.getImgIds(imgIds=[324158])
# print("-----imgIds--2", imgIds)

img = coco.loadImgs((imgIds[np.random.randint(0, len(imgIds))]))[0]
print("-----img", img)

I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
# plt.show()


# =========test3==========
# 加上segmentation标注信息
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print("-----annotations", anns)
coco.showAnns(anns)
# 显示加上标注信息的图片
plt.show()




