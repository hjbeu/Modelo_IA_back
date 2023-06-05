from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import torch as T
import torchvision.transforms as TT
import torch.nn as nn
import requests as req
from torchvision import datapoints
from torchvision.io import read_image
from pathlib import Path
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

app = FastAPI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/predict')
def fazer_predict(file: UploadFile):
    # Save file to temp dir
    with open('temp/'+ file.filename, 'wb') as buffer:
        buffer.write(file.file.read())
    result = predict(file.filename)  # Chama a função de previsão
    return result  # Converte o resultado para uma lista e retorna como JSON


# Carregar o modelo treinado
class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Dropout after third conv layer
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Dropout after fifth conv layer
            nn.MaxPool2d(2, 2)
            )

        self.linearlayers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout after first linear layer
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.convlayers(x)
        x = T.flatten(x, 1)
        x = self.linearlayers(x)
        return x

modelo = ConvolutionalModel()
modelo.load_state_dict(T.load("./model/v5-81.pth", map_location=T.device('cpu')))
def predict(input):
    input = pil_loader('temp/' + input)
    prep_transforms = TT.Compose(
            [TT.Resize((32, 32)),
            TT.ToTensor(),
            TT.Normalize( (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616) )
            ]
        )
    input = prep_transforms(input)
    output = modelo(input.unsqueeze(0))
    CATEGORIES = ['Avião','Carro','Passaro','Gato','Cervo',
               'Cachorro','Sapo','Cavalo','Navio','Caminhão']
    # Implemente a lógica de pós-processamento da saída aqui, se necessário
    logits = T.nn.functional.softmax(output, dim=1) * 100
    prob_dict = {}
    for i, classname in enumerate(CATEGORIES):
      prob = logits[0][i].item()
      print(f"{classname} score: {prob:.2f}")
      prob_dict[classname] = [prob]
    max_value, max_index = T.max(output, dim=1)
    max_index = CATEGORIES[max_index.item()]
    resultado = "Provavelmente é um " + max_index
    prob= prob_dict
    return {'prob': prob , 'resu': resultado}