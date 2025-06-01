import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from class_names import class_names
import os
import urllib.request

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_model_if_needed(path="resnet18_food101_top20_epoch3.pth"):
    if not os.path.exists(path):
        print("📥 模型不存在，開始下載...")
        url = "https://drive.google.com/uc?export=download&id=1vAA7vsaYzzg09F-Nt7TH7VvD9o2kFvHg"
        urllib.request.urlretrieve(url, path)
        print("✅ 模型下載完成")

def load_model(path="resnet18_food101_top20_epoch3.pth"):
    download_model_if_needed(path)
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    # 等待檔案確實下載完成（最多等 60 秒）
    for i in range(30):
        if os.path.exists(path):
            break
        print(f"等待模型檔案...({i+1})")
        time.sleep(2)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型檔案：{path}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)

def predict_image(model, image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]
