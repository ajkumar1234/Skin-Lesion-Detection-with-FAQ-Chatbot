import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=7, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def predict(model, image_tensor, device="cpu"):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
    return pred.item()
