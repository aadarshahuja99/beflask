import io
from torchvision import models
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import torch
classes=['LeafBlast', 'BrownSpot', 'Healthy', 'Hispa']
def get_model():
    checkpoint_path='model_transfer4.pt'
    model_transfer = models.googlenet(pretrained=True)
    for param in model_transfer.parameters():
        param.requires_grad=True
    
    n_inputs = model_transfer.fc.in_features 

    last_layer = nn.Linear(n_inputs, len(classes))

    model_transfer.fc = last_layer
    print(model_transfer.fc.out_features)
    model_transfer.load_state_dict(torch.load(checkpoint_path, map_location='cpu'),strict=False)
    model_transfer.eval()
    return model_transfer

def get_tensor(image_bytes) :
    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    image=Image.open(image_bytes)
    return test_transforms(image).unsqueeze(0)
