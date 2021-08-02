import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread("/home/fusion4268/4DShell/datasets/Postech2/1280/image_0/01111.png").astype(np.float32)
src = src/256
dst1 = src[163:557, 0:1280]
dst2 = cv2.resize(dst1, (832, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
imgplot = plt.imshow(dst2)
plt.savefig('out.png')

"""
rows,cols,_ = src.shape
for i in range(rows):
    for j in range(cols):
        k = src[i,j]
        print(k)

cv2.imshow("src", src)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
