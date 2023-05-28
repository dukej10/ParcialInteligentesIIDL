import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras import regularizers


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

model=Sequential()
#Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))  #Rearmar la imagen para que queden con la forma correspondiente

#Capas Ocultas
#Capas convolucionales
# Same dudplica los ultimos datos
model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name="capa_1", kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=2,strides=2)) #Reduccion de la imagen y quedarse con los datos mas caracteristicos

model.add(Conv2D(kernel_size=3,strides=1,filters=36,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

#Aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="relu", kernel_regularizer=regularizers.l2(0.01)))

#Capa de salida
model.add(Dense(numeroCategorias,activation="sigmoid"))


#Traducir de keras a tensorflow
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])

# Verificación de sobreajuste
# Detiene el entrenamiento del modelo antes de ocurra el entrenamiento
# Si no detecta una mejoría en la métrica
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)


#Entrenamiento
history = model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=60, validation_split=0.3, callbacks=[early_stopping])



#Prueba del modelo


imagenesPrueba,probabilidadesPrueba=cargarDatos("../dataset/test/grays/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=",resultados[1])

# Obtener las etiquetas predichas
y_pred = model.predict(x=imagenesPrueba)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(probabilidadesPrueba, axis=1)

print("============================")
print("     Métricas")
# Matriz de confusión
confusion = confusion_matrix(y_true, y_pred_classes)
print("Matriz de confusión:")
print(confusion)

# Accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_true, y_pred_classes, average='weighted')
print("Precision:", precision)

# Recall
recall = recall_score(y_true, y_pred_classes, average='weighted')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print("F1 Score:", f1)

# Pérdida
loss = resultados[0]
print("Loss:", loss)


# Guardar modelo
ruta="models/modeloA_red1.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()


# Gráficas de aprendizaje
print(history)
print(history.history)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()