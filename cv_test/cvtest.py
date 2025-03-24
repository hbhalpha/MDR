import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# 读取图像
image = cv2.imread(r"E:\MMIR\Challenge\Philip_Francis_Thomas.png" )
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 170], dtype=np.uint8)
upper_white = np.array([180, 60, 255], dtype=np.uint8)
mask_white = cv2.inRange(hsv, lower_white, upper_white)
mask_nonwhite = cv2.bitwise_not(mask_white)
# 6. 用该掩码保留“非白色”部分（把白色区域过滤掉）
image_no_white = cv2.bitwise_and(image, image, mask=mask_nonwhite)
gray = cv2.cvtColor(image_no_white, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 120)
mask = np.zeros_like(gray)  # 初始化掩码

kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=2)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 5000
saved_images = []

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    if 0.3 < aspect_ratio < 2:
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        cropped = image[y:y + h, x:x + w]
        saved_images.append(cropped)
for idx, cropped_image in enumerate(saved_images[:10]):
    output_image_path = os.path.join("./hahaha", f"{idx+1}.jpg")
    cv2.imwrite(output_image_path, cropped_image)

print(f"Processed , extracted {len(saved_images[:10])} images.")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"提取到 {len(saved_images)} 张图片。")
