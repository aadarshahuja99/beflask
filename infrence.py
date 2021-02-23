from commons import get_tensor, get_model
import torchvision
import torch
import numpy as np
model = get_model()
def get_class_name(image_bytes) :
    tensor=get_tensor(image_bytes)
    output= model(tensor)
    _,preds_tensor = torch.max(output,1)
    preds = np.squeeze(preds_tensor.numpy())
    classes=['LeafBlast', 'BrownSpot', 'Healthy', 'Hispa']
    category=classes[preds]
    return category
