
import cv2
from Prediccion import Prediccion
import numpy as np
from sklearn.metrics import confusion_matrix

from entrenamiento import numeroCategorias, cantidaDatosPruebas, cargarDatos

class PruebasCNN:
    def __init__(self):

        self.clases=["6","7","8","9","10","11","12"]
        self.valores ={
            "6": {"valor": 6, "clase": "6 de Bastos"}, "7": {"valor": 7, "clase": "7 de Bastos"}, "8":{"valor": 8,
            "clase": "8 de Bastos"}, "9": {"valor": 9, "clase": "9 de Bastos"},
          "10":{"valor": 10, "clase": "Sota de Bastos"}, "11":{"valor": 11, "clase": "Caballo de Bastos"},
        "12": {"valor": 12, "clase": "Rey de Bastos"},
        }
        self.ancho=128
        self.alto=128
        self.miModeloCNN=Prediccion("models/modeloA.h5",self.ancho,self.alto)



    def process(self):
        imagen = cv2.imread("contorno_0.jpg")
        claseResultado=self.miModeloCNN.predecir(imagen)
        print("La imagen cargada es ",self.valores[self.clases[claseResultado]])
        return self.valores[self.clases[claseResultado]]
"""
# ----- METRICAS -----------
# Carga de imágenes de prueba y etiquetas correspondientes
imagenesPrueba, etiquetasPrueba = cargarDatos("dataset/test/", numeroCategorias, cantidaDatosPruebas, ancho, alto)

# Realizar predicciones en imágenes de prueba
predicciones = miModeloCNN.modelo.predict(x=imagenesPrueba)

# Obtener etiquetas predichas
clasesPredichas = np.argmax(predicciones, axis=1)

etiquetasPrueba = np.argmax(etiquetasPrueba, axis=1)

# Calcular matriz de confusión
matrizConfusion = confusion_matrix(etiquetasPrueba, clasesPredichas)
print("Matriz de Confusión:")
print(matrizConfusion)

# -------------------------------------------------------------------

"""
