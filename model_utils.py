import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from class_names import class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path="resnet18_food101_top20_epoch3.pth"):
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
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
