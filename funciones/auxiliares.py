# -*- coding: utf-8 -*-
'''
Funciones auxiliares para red ViT:
'''
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import cv2 



#%%%%

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

from tqdm import tqdm, trange

import cv2

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)



from torch import Tensor
from einops import rearrange , reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from sklearn.model_selection import train_test_split

from sklearn import metrics as metsk
import matplotlib.pyplot as plt

from torchsummary import summary
#%%%





class Lector(Dataset):
    def __init__(self, data, modo,ruta_images, transform=None):
        self.data = data #df
        self.modo = modo # numero
        self.ruta_images = ruta_images
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # image = self.data[index]
        img_path = self.ruta_images+self.data['ruta'].iloc[index]
        image = cv2.imread(img_path)
        label = self.data[f'clases_{self.modo}'].iloc[index]

        # Aplicar transformaciones si se proporcionan
        if self.transform:
            image = self.transform(image)

        return image, label
    

def train_funcion(red,criterio,optimizador,ruta_guardado_red,ruta_guardado_optimizer,
                  dataloader_train, dataloader_val,titulo_grafico_loss,n_epoch):

  model = red
  criterion = criterio
  optimizer = optimizador
  train_loader = dataloader_train
  val_loader = dataloader_val
  N_EPOCHS = n_epoch
  loss_train_history=[]
  loss_val_history=[]
  for epoch in trange(N_EPOCHS, desc="Training"):
      correct, total = 0, 0
      train_loss = 0.0
      for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
          x, y = batch
          x, y = x.to(device), y.to(device)
          y_hat = model(x)
          # print(y_hat)
          loss = criterion(y_hat, y)
          train_loss += loss.detach().cpu().item() / len(train_loader)

          correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
          total += len(x)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      loss_train_history.append(train_loss)
      print(f"Train accuracy: {correct / total * 100:.2f}%")
      print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

      # Test loop
      with torch.no_grad():
          correct, total = 0, 0
          test_loss = 0.0
          for batch in tqdm(val_loader, desc="Testing in validation"):
              x, y = batch
              x, y = x.to(device), y.to(device)
              y_hat = model(x)
              loss = criterion(y_hat, y)
              test_loss += loss.detach().cpu().item() / len(test_loader)

              correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
              total += len(x)
          loss_val_history.append(test_loss)
          print(f"Val loss: {test_loss:.2f}")
          print(f"Val accuracy: {correct / total * 100:.2f}%")

  torch.save(model.state_dict(), ruta_guardado_red)
  torch.save(optimizer.state_dict(), ruta_guardado_optimizer)

  # Grafica las curvas de loss
  plt.plot(loss_train_history, label='Training Loss')
  plt.plot(loss_val_history, label='Validation Loss')
  plt.xlabel('Épocas')
  plt.ylabel('Loss')
  plt.title(titulo_grafico_loss)
  plt.legend()
  plt.show()

  from sklearn import metrics as metsk
import matplotlib.pyplot as plt
def plot_confusion_matrix(labels, pred_labels, classes,normalizacion, title="Confusion Matrix"):

    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = metsk.confusion_matrix(labels, pred_labels, normalize=normalizacion)
    disp =  metsk.ConfusionMatrixDisplay(cm, display_labels=classes)

    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(title)
    # Rotar las etiquetas del eje x en 45 grados
    plt.xticks(rotation=45, ha="right")
    fig.show()

def replace_labels_with_names(all_labels, all_preds, lista_clases):
    all_labels_names = [lista_clases[label] for label in all_labels]
    all_preds_names = [lista_clases[pred] for pred in all_preds]
    return all_labels_names, all_preds_names

def graficar_metricas(list_dat, ruta_mod, nombre_modelo, lista_clases):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT((3, 224, 224), n_patches=14, n_blocks=12, \
                hidden_d=768, n_heads=12, out_d=len(lista_clases)).to(device)
    network_state_dict = torch.load(ruta_mod)
    model.load_state_dict(network_state_dict)

    # Evaluar el modelo en el conjunto de prueba
    model.eval()

    for i, dataload in enumerate(list_dat):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataload:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Reemplazar números con nombres de clases en las etiquetas
        all_labels_names, all_preds_names = replace_labels_with_names(all_labels, all_preds, lista_clases)

        # Calcular y mostrar la matriz de confusión
        classes = lista_clases
        conjuntos_name = ["Conjunto de Train", "Conjunto de Val", "Conjunto de Test"]
        print(f"Metricas modelo : {nombre_modelo}, {conjuntos_name[i]}")

        print(metsk.classification_report(all_labels_names, all_preds_names, digits=4))

        plot_confusion_matrix(all_labels, all_preds, classes, 'true',
                              title=f"Confusion Matrix, Modelo {nombre_modelo}, {conjuntos_name[i]}")
