import os
import numpy as np
from sklearn.cluster import KMeans
import cv2
from imutils import build_montages
import matplotlib.image as imgplt

image_path = []
all_images = []
images = os.listdir('../Dataset/datas')

for image_name in images:
    image_path.append('../Dataset/datas/' + image_name)
for path in image_path:
    image = imgplt.imread(path)
    image = image.reshape(-1)
    all_images.append(image)

# 原来的图片是A~K,因此预期的分类是11类
clt = KMeans(n_clusters=11, max_iter=1000)
clt.fit(all_images)
labelIDs = np.unique(clt.labels_)
for labelID in labelIDs:
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)
    show_box = []
    for i in idxs:
        image = cv2.imread(image_path[i])
        image = cv2.resize(image, (96, 96))
        show_box.append(image)
    montage = build_montages(show_box, (96, 96), (5, 5))[0]

    title = "Type {}".format(labelID)
    cv2.imshow(title, montage)
    cv2.waitKey(0)
