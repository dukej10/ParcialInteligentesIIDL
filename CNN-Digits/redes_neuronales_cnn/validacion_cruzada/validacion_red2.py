from sklearn.metrics import confusion_matrix

# Número de folds para la validación cruzada
num_folds = 5

# Crear los conjuntos de entrenamiento y prueba mediante validación cruzada
kf = KFold(n_splits=num_folds, shuffle=True)

# Listas para almacenar los resultados de cada fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
losses = []
confusion_matrices = []

# Iterar sobre los folds
for train_index, test_index in kf.split(imagenes):
    # Obtener los conjuntos de entrenamiento y prueba para el fold actual
    x_train, x_test = imagenes[train_index], imagenes[test_index]
    y_train, y_test = probabilidades[train_index], probabilidades[test_index]

    # Crear y compilar el modelo
    model = Sequential()
    model.add(InputLayer(input_shape=(pixeles,)))
    model.add(Reshape(formaImagen))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=numeroCategorias, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Entrenar el modelo
    history = model.fit(x=x_train, y=y_train, epochs=30, batch_size=60, callbacks=[early_stopping])

    # Evaluar el modelo en el conjunto de prueba del fold actual
    results = model.evaluate(x=x_test, y=y_test)

    # Obtener las métricas de evaluación
    accuracy = results[1]
    precision = precision_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1), average='weighted')
    recall = recall_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1), average='weighted')
    f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1), average='weighted')
    loss = results[0]

    # Almacenar los resultados del fold actual
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    losses.append(loss)

    # Obtener la matriz de confusión del fold actual
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    confusion_matrices.append(cm)

# Calcular el promedio de las métricas de evaluación en todos los folds
mean_accuracy = np.mean(accuracies)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1 = np.mean(f1_scores)
mean_loss = np.mean(losses)

# Imprimir los resultados promedio
print("Promedio Accuracy:", mean_accuracy)
print("Promedio Precision:", mean_precision)
print("Promedio Recall:", mean_recall)
print("Promedio F1 Score:", mean_f1)
print("Promedio Loss:", mean_loss)

# Calcular la matriz de confusión promedio
avg_cm = np.mean(confusion_matrices, axis=0)
print("Matriz de Confusión Promedio:")
print(avg_cm)
