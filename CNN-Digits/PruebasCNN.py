
import cv2
from Prediccion import Prediccion
import numpy as np
from sklearn.metrics import confusion_matrix


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



    def process(self, numeroPredicciones):
        imagen1 = None
        imagen2 = None
        datos = []
        if numeroPredicciones == 2:
            imagen1 = cv2.imread("imgs/dos/primer_cuadrado.jpg")
            imagen2 = cv2.imread("imgs/dos/segundo_cuadrado.jpg")
            if imagen2 is not None and imagen1 is not None:
                claseResultado = self.miModeloCNN.predecir(imagen1)
                datos.append(self.valores[self.clases[claseResultado]])
                claseResultado = self.miModeloCNN.predecir(imagen2)
                datos.append(self.valores[self.clases[claseResultado]])
                print(datos[0])
                print(datos[1])
               
            else:
                datos.append({"mensaje": ""})
                

        else:
            imagen = cv2.imread("imgs/una/primer_cuadrado.jpg")
            claseResultado=self.miModeloCNN.predecir(imagen)
            datos.append(self.valores[self.clases[claseResultado]])
            print("La imagen cargada es", datos)
        return datos
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
