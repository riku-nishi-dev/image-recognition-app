import cv2
import numpy as np
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self,num_classes: int):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(16,128,kernel_size=3,padding=1),
            nn.ReLU(),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Dropout(0.5)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*7*7,128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x
    
class TorchRecognizer:
    def __init__(
            self,
            model_path: str,
            labels: list[str]|None=None,
            image_size: tuple[int,int]=(28,28),
            device: str|None=None,
    ):
        if labels is None:
            labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "blank"]
        
        self.labels=labels
        self.image_width,self.image_height=image_size

        if device is None:
            device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=torch.device(device)

        self.model=CNNModel(num_classes=len(self.labels)).to(self.device)
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model.eval()

    def preprocess(self,roi_bgr: np.ndarray)->torch.Tensor:
        gray=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,(self.image_width,self.image_height))
        x=gray.astype("float32")/255.0

        tensor=torch.tensor(x,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_digit(self,roi_bgr: np.ndarray)->tuple[str,float]:
        input_tensor=self.preprocess(roi_bgr)

        with torch.no_grad():
            output=self.model(input_tensor)
            probs=torch.softmax(output,dim=1)
            score,pred=torch.max(probs,dim=1)
        
        value=self.labels[pred.item()]
        confidence=float(score.item())
        return value,confidence
    
