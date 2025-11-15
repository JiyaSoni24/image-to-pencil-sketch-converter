import cv2
import matplotlib.pyplot as plt

image = cv2.imread('closeup-shot-young-tiger-resting-piece-wood.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
inverted_gray = cv2.bitwise_not(gray_image)
blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
inverted_blur = cv2.bitwise_not(blurred)
sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sketch, cmap='gray')
plt.title("Pencil Sketch")
plt.axis('off')

plt.show()

cv2.imwrite('pencil_sketch.jpg', sketch)
