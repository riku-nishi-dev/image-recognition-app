import cv2
from .digit_detect import detect_digit_bboxes

def infer_image(img_bgr,recognizer,expected_digits=3,min_score=0.7,debug=False):
    img_bgr=cv2.resize(img_bgr,(400,100))
    bboxes=detect_digit_bboxes(img_bgr,expected_digits=expected_digits)
    if len(bboxes)==0:
        return {
            "ok":False,
            "meter_value":"",
            "reason":"no_digits_detected",
            "digits":[]
        }
    digits=[]
    for (x,y,w,h) in bboxes:
        roi=img_bgr[y:y+h,x:x+w]
        value,score=recognizer.predict_digit(roi)

        digits.append({
            "bbox":[int(x),int(y),int(w),int(h)],
            "value":value,
            "score":float(score)
        })
    digits=sorted(digits,key=lambda d: d["bbox"][0])

    filtered_values=[d["value"] for d in digits if d["value"]!="blank"]
    meter_value="".join(filtered_values)

    ok=True
    reason=None

    if len(digits)<expected_digits:
        ok=False
        reason="too_few_digits"
    elif min(d["score"] for d in digits)<min_score:
        ok=False
        reason="low_confidence"
    return {
        "ok":ok,
        "meter_vlaue":meter_value,
        "digits":digits
    }

