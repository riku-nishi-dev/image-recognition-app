import argparse
import glob
import json
import os
import cv2

from meter_reader.recognize_torch import TorchRecognizer
from meter_reader.pipeline import infer_image

def iter_images(path):
    if os.path.isdir(path):
        files=[]
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            files.extend(glob(os.path.join(path,ext)))
        return sorted(files)
    return [path]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,help="image file or directory")
    parser.add_argument("--model",default="models/model.pth")
    parser.add_argument("--expected-digits",type=int,default=3)
    parser.add_argument("--min-score",type=float,default=0.7)
    parser.add_argument("--out",default="outputs/results.jsonl")

    args=parser.parse_args()

    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    recognizer=TorchRecognizer(args.model)

    with open(args.out,"a",encoding="utf-8") as f:
        for fp in iter_images(args.input):
            img=cv2.imread(fp)
            if img is None:
                rec={"file":fp,"ok":False,"reason":"imread_failed"}
            else:
                rec=infer_image(
                    img,
                    recognizer,
                    expected_digits=args.expected_digits,
                    min_score=args.min_score
                )
                rec["file"]=fp
            f.write(json.dumps(rec,ensure_ascii=False)+"\n")
            print(fp, rec.get("meter_value",""), "OK" if rec.get("ok") else f"NG({rec.get('reason')})")

if __name__=="__main__":
    main()

