# Parcial Inteligentes II
Ver Rama final
Proyecto que permite a través de inteligencia artificial y redes neuronales convusionales detectar las cartas de bastones del 6 al 11 de la baraja española, permitiendo detectar dos cartas o 1 e ir sumando su valor correspondiente 

## Estudiantes: Juan Diego Duque López - Alejandro Trujillo

Dataset: El Dataset empleado para el entrenamiento y prueba de los modelos de redes neuronales se encuentra en la carpeta dataset/train/grays y datasert/test/grays

Imágenes a predecir: estás imágenes quedan en la carpeta img, cuando se van a predecir dos imágenes, las dos imágenes recortadas quedan en la carpeta dos, y cuando se va a predecir una carta, la imagen recortada queda en la carpeta una

## Resultados de los modelos

Matrices de Confusión

a. ModeloR1

![Matriz de confusion modeloR1](https://drive.google.com/uc?export=view&id=1LZra-x50Yy6JyNREMDTg8oxfWYvrCOEY)

b. ModeloR2

![Matriz de confusion modeloR2](https://drive.google.com/uc?export=view&id=1EMV6KRJz5DnCmm-rC0NJgB0kMOkJmvrn)

c. ModeloR3

![Matriz de confusion modeloR3](https://drive.google.com/uc?export=view&id=1Hg-gv1Lfd-fhhKdoEf8sI9ZbzuBCk3M4)


Se realizó la implementación y entrenamiento de 3 modelos de redes neuronales los cuales se encuentran dentro de la carpeta redes_neuronales_cnn y los resultados de sus pruebas fueron los siguientes:
| N° | Nombre modelo | Accuracy | Precision | Recall | F1 Score | Loss | Épocas de entrenamiento | Tiempo de respuesta |
|----|---------------|---------------------|---------------------|---------------------|---------------------|---------------------|------------------------|----------------------|
| 1 | ModelR1 | 0.9630051223676722 | 0.9702151782698896 | 0.9630051223676722 | 0.962409429392541 | 0.0021570068784058094| 30 | 1.555873155593872 |
| 2 | ModelR2 | 0.9535201640464799 | 0.9649303764827231 | 0.9535201640464799 | 0.9522566557487192 | 0.0014912157785147429| 30 | 1.2763197422027588 |
| 3 | ModelR3 | 0.7638019351166762 | 0.8507024593958814 | 0.7638019351166762 | 0.7566304928679441 | 0.8830338716506958 | 35 | 2.096485137939453 |

## Video:

A continuación encontrará el link del video con una explicación sobre el proyecto
https://www.youtube.com/watch?v=gPO-AjMOvAQ
