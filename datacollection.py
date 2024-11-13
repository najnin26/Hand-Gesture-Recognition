import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "Data/Okay"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the crop region is within image bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:  # Check if the crop has content
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Step 1: Convert to Grayscale
            gray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Grayscale', gray)

            # Step 2: Binarization
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow('Binarized', binary)

            # Step 3: Noise Removal using Median Filter
            denoised = cv2.medianBlur(binary, 3)
            cv2.imshow('Denoised', denoised)

            # Step 4: Remove Small Components (BLOB Processing)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(denoised, connectivity=8)
            sizes = stats[1:, -1]  # Skip background label
            min_size = 500  # Minimum size of blob to keep

            filtered_img = np.zeros_like(denoised)
            for i in range(1, num_labels):
                if sizes[i - 1] >= min_size:
                    filtered_img[labels == i] = 255
            cv2.imshow('Blob Removed', filtered_img)

            # Set `imgWhite` to the output after blob removal
            imgWhite = filtered_img

            # Uncomment if you want to view the intermediate output for blob removal
            # cv2.imshow('ImageWhite', imgWhite)

            cv2.imshow('ImageCrop', imgCrop)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

cap.release()
cv2.destroyAllWindows()
