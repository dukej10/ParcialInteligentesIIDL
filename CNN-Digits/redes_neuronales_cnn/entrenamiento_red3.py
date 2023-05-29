import tensorflow as tf
import keras
import numpy as np
import cv2
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
import seaborn as sns


###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
##################################

def cargarDatos(rutaOrigen,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=rutaOrigen+str(categoria+6)+"/"+str(categoria+6)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) #     Convertir imagena a escala de grises
            imagen = cv2.resize(imagen, (ancho, alto))    # Redimensionar la imagen
            imagen = imagen.flatten()   # Pasar de matriz a vector
            imagen = imagen / 255  # Dejar los valores entre 0 y 1
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias) # arrancar en ceros
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

#################################
# Dimensiones de la imagenes a trabajar - Las imagenes a usar deben tener el mismo tamaño
ancho=128
alto=128
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=1  # Para trabajar con escala de grises debe ser 1
formaImagen=(ancho,alto,numeroCanales)
numeroCategorias=7  # son 10 por se clasificaran en 10 categorias según el numero

cantidaDatosEntrenamiento=[209,209,209,209,209,209,209] # Cuantas imagenes hay para el entrenamiento por cada clase
cantidaDatosPruebas=[209,209,209,209,209,209,209] # Cuantas imagenes hay para la prueba por cada clase

#Cargar las imágenes
imagenes, probabilidades=cargarDatos("../dataset/train/grays/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)

modelR3=Sequential()
#Capa entrada
modelR3.add(InputLayer(input_shape=(pixeles,)))
modelR3.add(Reshape(formaImagen))  #Rearmar la imagen para que queden con la forma correspondiente

#Capas Ocultas
#Capas convolucionales
# Same dudplica los ultimos datos
modelR3.add(Conv2D(kernel_size=5, strides=2, filters=16, padding="same", activation="relu", name="capa_1"))
modelR3.add(MaxPool2D(pool_size=2, strides=2)) #Reduccion de la imagen y quedarse con los datos mas caracteristicos

modelR3.add(Conv2D(kernel_size=3, strides=1, filters=36, padding="same", activation="relu", name="capa_2"))
modelR3.add(MaxPool2D(pool_size=2, strides=2))

#Aplanamiento
modelR3.add(Flatten())
modelR3.add(Dense(128, activation="softmax"))

#Capa de salida
modelR3.add(Dense(numeroCategorias, activation="softmax"))


#Traducir de keras a tensorflow
modelR3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#Entrenamiento
# Verificación de sobreajuste
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Entrenamiento con conjunto de validación
history = modelR3.fit(x=imagenes, y=probabilidades, epochs=25, batch_size=60, callbacks=[early_stopping])

#Prueba del modelo
start_time = time.time()  # Obtener el tiempo de inicio
imagenesPrueba,probabilidadesPrueba=cargarDatos("../dataset/test/grays/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados=modelR3.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
end_time = time.time()  # Obtener el tiempo de finalización
elapsed_time = end_time - start_time  # Calcular el tiempo transcurrido
print("Tiempo de ejecución:", elapsed_time)
print("Accuracy=",resultados[1])

# Evaluación de overfitting
print("<===============>")
print("Verificación Overfitting")
resultadosEntrenamiento = modelR3.evaluate(x=imagenes, y=probabilidades)
print("Training Loss:", resultadosEntrenamiento[0])
print("Training Accuracy:", resultadosEntrenamiento[1])
print("Validation Loss:", resultados[0])
print("Validation Accuracy:", resultados[1])

#Métricas
print("============================")
print("     Métricas")

# Obtener las etiquetas predichas
y_pred = modelR3.predict(x=imagenesPrueba)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(probabilidadesPrueba, axis=1)

# Matriz de confusión
confusion = confusion_matrix(y_true, y_pred_classes)
print("Matriz de confusión:")
print(confusion)

# Accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=1)
print("Precision:", precision)

# Recall
recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=1)
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=1)
print("F1 Score:", f1)

# Pérdida
loss = resultados[0]
print("Loss:", loss)
# Guardar modelo
ruta="../models/modeloA_red3.h5"
modelR3.save(ruta)
# Informe de estructura de la red
modelR3.summary()

# Matriz de confusión gráfica

labels = ["6", "7", "8", "9", "10", "11", "12"]

# Crear una figura de matplotlib
fig, ax = plt.subplots(figsize=(8, 6))

# Utilizar seaborn para crear un mapa de calor de la matriz de confusión
sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d", cbar=False)
# Configurar etiquetas de los ejes
ax.set_xlabel("Predicciones")
ax.set_ylabel("Valores verdaderos")
ax.set_title("Matriz de Confusión")
# Configurar etiquetas personalizadas en los ejes x e y
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Mostrar la figura
plt.show()