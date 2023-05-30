import tensorflow as tf
import keras
import numpy as np
import cv2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer, Input, Conv2D, MaxPool2D, Reshape, Dense, Flatten
from sklearn.model_selection import StratifiedKFold, cross_val_score


##################################

def cargarDatos(rutaOrigen, numeroCategorias, limite, ancho, alto):
    imagenesCargadas = []
    valorEsperado = []
    for categoria in range(0, numeroCategorias):
        for idImagen in range(0, limite[categoria]):
            ruta = rutaOrigen + str(categoria + 6) + "/" + str(categoria + 6) + "_" + str(idImagen) + ".jpg"
            print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir imagena a escala de grises
            imagen = cv2.resize(imagen, (ancho, alto))  # Redimensionar la imagen
            imagen = imagen.flatten()  # Pasar de matriz a vector
            imagen = imagen / 255  # Dejar los valores entre 0 y 1
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)  # arrancar en ceros
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados


#################################
# Dimensiones de la imagenes a trabajar - Las imagenes a usar deben tener el mismo tamaño
ancho = 128
alto = 128
pixeles = ancho * alto
# Imagen RGB -->3
numeroCanales = 1  # Para trabajar con escala de grises debe ser 1
formaImagen = (ancho, alto, numeroCanales)
numeroCategorias = 7  # son 10 por se clasificaran en 10 categorias según el numero

cantidaDatosEntrenamiento = [251, 251, 251, 251, 251, 251,
                             251]  # Cuantas imagenes hay para el entrenamiento por cada clase
cantidaDatosPruebas = [251, 251, 251, 251, 251, 251, 251]  # Cuantas imagenes hay para la prueba por cada clase

# Cargar las imágenes
imagenes, probabilidades = cargarDatos("../../dataset/train/grays/", numeroCategorias, cantidaDatosEntrenamiento, ancho,
                                       alto)

modelR1 = Sequential()
# Capa entrada
modelR1.add(InputLayer(input_shape=(pixeles,)))
modelR1.add(Reshape(formaImagen))  # Rearmar la imagen para que queden con la forma correspondiente

# Capas Ocultas
# Capas convolucionales
# Same dudplica los ultimos datos
modelR1.add(Conv2D(kernel_size=5, strides=2, filters=16, padding="same", activation="relu", name="capa_1"))
modelR1.add(MaxPool2D(pool_size=2, strides=2))  # Reduccion de la imagen y quedarse con los datos mas caracteristicos

modelR1.add(Conv2D(kernel_size=3, strides=1, filters=36, padding="same", activation="relu", name="capa_2"))
modelR1.add(MaxPool2D(pool_size=2, strides=2))

# Aplanamiento
modelR1.add(Flatten())
modelR1.add(Dense(128, activation="relu"))

# Capa de salida
modelR1.add(Dense(numeroCategorias, activation="sigmoid"))

# Traducir de keras a tensorflow
modelR1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# Entrenamiento
# Verificación de sobreajuste
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Entrenamiento con conjunto de validación
# history = modelR1.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=60, callbacks=[early_stopping])

# Validacion cruzada
numero_fold = 1
accuracy_fold = []
loss_fold = []

kFold = KFold(n_splits=5, shuffle=True)

for train, test in kFold.split(imagenes, probabilidades):
    print("##################Training fold ", numero_fold, "###################################")
    history = modelR1.fit(imagenes[train], probabilidades[train], epochs=30, batch_size=60)
    # Epochs --> Cantidad de veces que debe repetir el entrenamiento
    # Batch --> Cantidad de datos que puede cargar en memoria para realizar el entrenamiento en una fase
    metricas = modelR1.evaluate(imagenes[test], probabilidades[test])
    accuracy_fold.append(metricas[1])
    loss_fold.append(metricas[0])
    numero_fold += 1

for i in range(0,len(loss_fold)):
  print("Fold ",(i+1),"- Loss(Error)=",loss_fold[i]," - Accuracy=",accuracy_fold[i])
print("Average scores")
print("Loss",np.mean(loss_fold))
print("Accuracy",np.mean(accuracy_fold))