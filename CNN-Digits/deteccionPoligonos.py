import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

from PruebasCNN import PruebasCNN
pruebasCNN = PruebasCNN()
nameWindow = "Calculadora"
#Controlar si ya se realizó la prediccion
prediccion = False
# Almacenar el resultado de la prediccion
mensaje = []
# Almacenar los valores y nombres de cada carta
mensaje1 = {'clase': ''}
mensaje2 = {'clase': ''}
# Texto a mostrar en pantalla
procesar_text = ""
procesar_text2 = ""
restar_text = ""
opcion_text = ""
mode_text = ""
text_aux = ""
# Controlar si se deben mostrar los modos de detección
show_opciones = True
# Almacenará cuantas cartas se van a predecir al tiempo
n_cartas = 0
# controlar la ventana de video
mostrar_video = False
# Contar cuantos cuadrado se han detectado
num_squares = 0

# Controlar la captura de la foto
capture_photo = False

# Controlar el conteo de cuadrados
start_counting = False

# Almacenar los vertices de los cuadrados detectados
squares_vertices = []

# Almacena cuanto han sumado las cartas predecidas
total = 0
total_text = ""


#Almacena cuanto suman las cartas que se predijeron
acumulado = 0
acumulado_text = ""



ancho_deseado = 200
alto_deseado = 200
def nothing(x):
    pass



def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min", nameWindow, 0, 255, nothing)
    cv2.createTrackbar("max", nameWindow, 202, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 65, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 4500, 10000, nothing)

def calcularAreas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas


def detectarFigura(imagenOriginal):
    global num_squares, squares_vertices
    global start_counting
    imagenGris = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Gris", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gris", 600, 750)
    cv2.imshow("Gris", imagenGris)
    min_cv = cv2.getTrackbarPos("min", nameWindow)
    max_cv = cv2.getTrackbarPos("max", nameWindow)
    bordes = cv2.Canny(imagenGris, min_cv, max_cv)
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)
    i = 0
    areaMin = cv2.getTrackbarPos("areaMin", nameWindow)
    for figuraActual in figuras:
        if areas[i] >= areaMin:
            vertices = cv2.approxPolyDP(figuraActual, 0.05 * cv2.arcLength(figuraActual, True), True)
            if len(vertices) == 4:
                cv2.putText(imagenOriginal, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(imagenOriginal, text_aux, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                # Texto nombres de cartas detectadas
                cv2.putText(imagenOriginal, mensaje2["clase"], (300, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(imagenOriginal, mensaje1["clase"], (10, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                # Texto valor total de las cartas recien predecidas
                cv2.putText(imagenOriginal, total_text, (400, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                # Texto valor acumulado de todas las cartas predecidas
                cv2.putText(imagenOriginal, acumulado_text, (400, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Texto de indicaciones de pasos
                cv2.putText(imagenOriginal, procesar_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(imagenOriginal, procesar_text2, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(imagenOriginal, restar_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.drawContours(imagenOriginal, [figuraActual], 0, (0, 0, 255), 2)
                if start_counting:
                    squares_vertices.append(vertices.tolist())
                    if n_cartas == 2:  # Si el modo elegido es detección de dos cartas
                        num_squares += 1
                        if num_squares == 2:  # Si ya se detectaron dos cuadrados
                            # Obtener los vertices correspondientes de cada cuadro
                            vertices1 = squares_vertices[0]
                            vertices2 = squares_vertices[1]
                            # Coordenadas primer cuadrado
                            x1, y1 = vertices1[0][0]  # Coordenadas del primer vértice del primer cuadrado
                            x2, y2 = vertices1[1][0]  # Coordenadas del segundo vértice del primer cuadrado
                            x3, y3 = vertices1[2][0]  # Coordenadas del tercer vértice del primer cuadrado
                            x4, y4 = vertices1[3][0]  # Coordenadas del cuarto vértice del primer cuadrado
                            # coordenadas segundo cuadrado
                            x5, y5 = vertices2[0][0]  # Coordenadas del primer vértice del segundo cuadrado
                            x6, y6 = vertices2[1][0]  # Coordenadas del segundo vértice del segundo cuadrado
                            x7, y7 = vertices2[2][0]  # Coordenadas del tercer vértice del segundo cuadrado
                            x8, y8 = vertices2[3][0]  # Coordenadas del cuarto vértice del segundo cuadrado
                            # Obtener ROIS de la imagen
                            primer_cuadrado = imagenGris[min(y1, y2, y3, y4):max(y1, y2, y3, y4),
                                              min(x1, x2, x3, x4):max(x1, x2, x3, x4)]
                            segundo_cuadrado = imagenGris[min(y5, y6, y7, y8):max(y5, y6, y7, y8),
                                               min(x5, x6, x7, x8):max(x5, x6, x7, x8)]
                            # Redimensionar contenido de los cuadrados para que sea más fácil comparar
                            primer_cuadrado = cv2.resize(primer_cuadrado, (ancho_deseado, alto_deseado))
                            segundo_cuadrado = cv2.resize(segundo_cuadrado, (ancho_deseado, alto_deseado))
                            if primer_cuadrado is not None and segundo_cuadrado is not None:
                                # Comparar las imágenes de los cuadrados utilizando SSIM
                                similarity = ssim(primer_cuadrado, segundo_cuadrado)
                                similarity_threshold = 0.4  # umbral de similitud permitido
                                if similarity > similarity_threshold: # Verificar el contenido de los cuadrados
                                    print("Los cuadrados son similares. No se guardarán las imágenes.")
                                    squares_vertices = []
                                    num_squares = 0
                                else:
                                    # Guardar las imágenes de los cuadrados en archivos separados
                                    cv2.imwrite("imgs/dos/primer_carta.jpg", primer_cuadrado)
                                    recorte("imgs/dos/primer_carta.jpg")
                                    cv2.imwrite("imgs/dos/segundo_carta.jpg", segundo_cuadrado)
                                    recorte("imgs/dos/segundo_carta.jpg")
                                    # Para continuar detectando cuadrados se reinicia los párametros
                                    start_counting = False
                                    squares_vertices = []
                                    num_squares = 0
                            else:
                                squares_vertices = []
                                num_squares = 0
                    elif n_cartas == 1: #
                        vertices1 = squares_vertices[0]  # Coordenadas del primer cuadro
                        #  Extraer cada coordenada de los vértices
                        x1 = vertices1[0][0][0]
                        y1 = vertices1[0][0][1]
                        x2 = vertices1[1][0][0]
                        y2 = vertices1[1][0][1]
                        x3 = vertices1[2][0][0]
                        y3 = vertices1[2][0][1]
                        x4 = vertices1[3][0][0]
                        y4 = vertices1[3][0][1]
                        # Obtener ROI de la imagen
                        carta = imagenGris[min(y1, y2, y3, y4):max(y1, y2, y3, y4),
                                min(x1, x2, x3, x4):max(x1, x2, x3, x4)]
                        # Guardar
                        cv2.imwrite("imgs/una/carta.jpg", carta)
                        #recorte("imgs/una/carta.jpg")
                        # Reiniciar párametros para continuar detectando cartas
                        squares_vertices = []
                        start_counting = False
                else:
                    continue
        i += 1
    return imagenOriginal


def recorte(ruta):
    # Cargar la imagen
    img = cv2.imread(ruta)
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para obtener una imagen binaria
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre cada contorno y crear una imagen para cada uno
    for i, contour in enumerate(contours):
        # Crear una imagen en blanco del mismo tamaño que la imagen original
        contour_image = np.zeros_like(img)

        # Dibujar el contorno en la imagen en blanco
        cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Recortar la región dentro del contorno en la imagen original
        x, y, w, h = cv2.boundingRect(contour)
        cropped = img[y:y + h, x:x + w]

        # Guardar la imagen recortada para el contorno actual
        cv2.imwrite(ruta, cropped)


video = cv2.VideoCapture(1)
constructorVentana()

while True:
    _, frame = video.read()
    k = cv2.waitKey(5)
    if capture_photo and start_counting is False: # Continuar visualizando el video mientras se tomó la foto
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)  # Y permite actualizar el texto de instrucciones en Imagen
        cv2.resizeWindow("Video", 600, 750)
        mostrar_video = True
        cv2.imshow("Video", frame)
    else:
        frame = detectarFigura(frame)
        if show_opciones: # mostrar opciones de modo disponible para predecir
             procesar_text = 'Pulse 1 para predecir individualmente'
             procesar_text2 = 'Pulse 2 para predecir dos cartas'
             mode_text = "Elija opción de predicción"
        else:
             procesar_text = 'Pulse C para tomar la foto'
             procesar_text2 = ''
             total_text = f'La suma es {total}'
             acumulado_text = f'Acumulado {acumulado}'
        cv2.namedWindow("Imagen", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Imagen", 600, 750)
        cv2.imshow("Imagen", frame)
    if k == ord('1'): # El usuario seleccionó detección de a una carta
        show_opciones = False # Se oculta porque se seleccionó un modo
        n_cartas = 1
        mode_text = "Modo: Predecir una carta"
        procesar_text2 = ''
    elif k == ord('2'):  # El usuario seleccionó detección de dos cartas al tiempo
        show_opciones = False
        procesar_text2 = ''
        mode_text = "Modo: Predecir dos cartas"
        n_cartas = 2
    elif k == ord('q') or k == ord('Q'):  # Cerrar el programa
        break
    if show_opciones is False:
        if k == ord('c') or k == ord('C'): # Capturar la foto
            procesar_text = 'Pulse P para procesar la foto'
            restar_text = 'Pulse R para reiniciar'
            capture_photo = True
            start_counting = True
            prediccion = False
        elif k == ord('p') or k == ord('P'): # Procesar la imagen que se tomo para realizar la prediccion
            if prediccion is False: # Comprobar que no se haya predecido la carta actual
                text_aux = "Ultima(s) carta(s):"
                restar_text = ""
                mensaje = pruebasCNN.process(n_cartas) # obtiene el nombre de las cartas predecidad con su valor
                if len(mensaje) == 2: # Verifica si el modo es detección de dos cartas al tiempi
                   mensaje1 = mensaje[0]
                   mensaje2 = mensaje[1]
                   total = mensaje1["valor"] + mensaje2["valor"] # valor sumado de cartas predecidas
                   acumulado += total
                else:
                   mensaje1 = mensaje[0]
                   mensaje2 = {'clase': ''}
                   total= mensaje1["valor"] # valor de la carta predecida
                   acumulado+= total
                prediccion = True  # Ya se obtuvo la prediccion
                print(f'se detectó {mensaje}')
                capture_photo = False   # habilitar opción para tomar foto
                if mostrar_video:  # Verificar si ya existe la ventana
                    cv2.destroyWindow("Video")
                    mostrar_video = False
    else:
        mode_text = "Debe seleccionar un modo"
    if k == ord('r') or k == ord('R'):  # Continuar tomando fotos
        show_opciones = True
        restar_text = ""
        capture_photo = False
        prediccion = False
        if mostrar_video: # Verificar si ya existe la ventana
            cv2.destroyWindow("Video")
            mostrar_video = False
        total_text = ""
        acumulado_text = ""
        mensaje2 = {"clase":""}
        mensaje1 = {"clase": ""}



video.release()
cv2.destroyAllWindows()
