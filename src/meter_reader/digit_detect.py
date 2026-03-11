import cv2
import numpy as np
from imutils import contours

def _detect_by_otsu(img_bgr):
    img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img_gray=clahe.apply(img_gray)

    _,img_binary=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel=np.ones((3,3),np.uint8)
    img_dilation=cv2.morphologyEx(img_binary,cv2.MORPH_OPEN,kernel,iterations=1)
    img_dilation=cv2.dilate(img_dilation,kernel,iterations=3)

    return img_dilation

def _detect_by_adaptive(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_binary = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    img_dilation = cv2.dilate(img_dilation, kernel, iterations=3)

    return img_dilation


def _extract_bboxes(mask):
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    result = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)

        if not (30 < w < 150):
            continue
        if not (50 < h < 150):
            continue

        result.append([x, y, w, h])

    return result


def detect_digit_bboxes(img_bgr, expected_digits=3):
    bboxes1 = _extract_bboxes(_detect_by_otsu(img_bgr))

    if len(bboxes1) < expected_digits:
        bboxes2 = _extract_bboxes(_detect_by_adaptive(img_bgr))
        if len(bboxes2) > len(bboxes1):
            return sorted(bboxes2, key=lambda b: b[0])

    return sorted(bboxes1, key=lambda b: b[0])
