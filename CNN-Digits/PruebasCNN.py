
import cv2
from Prediccion import Prediccion
import numpy as np
from sklearn.metrics import confusion_matrix


class PruebasCNN:
    def __init__(self):
        # Información de las cartas
        self.clases=["6","7","8","9","10","11","12"]
        self.valores ={
            "6": {"valor": 6, "clase": "6 de Bastos"}, "7": {"valor": 7, "clase": "7 de Bastos"}, "8":{"valor": 8,
            "clase": "8 de Bastos"}, "9": {"valor": 9, "clase": "9 de Bastos"},
          "10":{"valor": 10, "clase": "Sota de Bastos"}, "11":{"valor": 11, "clase": "Caballo de Bastos"},
        "12": {"valor": 12, "clase": "Rey de Bastos"},
        }
        # Tamaño de las imágenes
        self.ancho=128
        self.alto=128
        # obtener el modelo
        self.miModeloCNN = Prediccion("models/modeloA_red1.h5", self.ancho, self.alto)


    # Método encargado de obtener las imágenes que se van a predecir, invocar la predicion para estas
    # y retornar el resultado de la prediccion
    def process(self, numeroPredicciones):
        # Almacenamiento de las imagenes
        imagen1 = None
        imagen2 = None
        # resultados de las predicciones
        datos = []
        if numeroPredicciones == 2:  # Si se van a predecir dos imagenes
            imagen1 = cv2.imread("imgs/dos/primer_carta.jpg")
            imagen2 = cv2.imread("imgs/dos/segundo_carta.jpg")
            if imagen2 is not None and imagen1 is not None:
                # Prediccion para la primera imagen
                claseResultado = self.miModeloCNN.predecir(imagen1)
                datos.append(self.valores[self.clases[claseResultado]])
                # Prediccion para la segunda imagen
                claseResultado = self.miModeloCNN.predecir(imagen2)
                datos.append(self.valores[self.clases[claseResultado]])
                print(datos[0])
                print(datos[1])
            else: # Si no se obtuvieron las imágenes
                datos.append({"mensaje": ""})
        else:   # Si se va a predecir una imagen
            imagen = cv2.imread("imgs/una/carta.jpg")
            # Prediccion de una imagen
            claseResultado=self.miModeloCNN.predecir(imagen)
            datos.append(self.valores[self.clases[claseResultado]])
            print("La imagen cargada es", datos)
        return datos