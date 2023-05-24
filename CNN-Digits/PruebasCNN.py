
import cv2
from Prediccion import Prediccion
import numpy as np
from sklearn.metrics import confusion_matrix

from entrenamiento import numeroCategorias, cantidaDatosPruebas, cargarDatos

clases=["6 Bastos","7 de Bastos","8 de Bastos","9 de Bastos","Sota de Bastos","Caballo de Bastos","Rey de Bastos"]

ancho=128
alto=128

miModeloCNN=Prediccion("models/modeloA.h5",ancho,alto)
imagen=cv2.imread("dataset/test/7/7_17.jpg")



claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])


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


while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()