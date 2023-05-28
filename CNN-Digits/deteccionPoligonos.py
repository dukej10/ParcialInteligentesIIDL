import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

from PruebasCNN import PruebasCNN
pruebasCNN = PruebasCNN()
nameWindow = "Calculadora"
prediccion = False
string = ""
mensaje = []
mensaje1 = {'clase': ''}
mensaje2 = {'clase': ''}
procesar_text = ""
procesar_text2 = ""
restar_text = ""
opcion_text = ""
show_opciones = True
n_cartas = 0
total_text = ""
mode_text = ""
text_aux = ""

# Variable to keep track of whether two squares have been detected
num_squares = 0

# Variable to control capturing the photo
capture_photo = False

# Variable to control counting squares
start_counting = False

# List to store the processed square vertices
processed_squares = []

# List to store the detected square vertices
squares_vertices = []

# Almacena cuanto han sumado las cartas
total = 0

ancho_deseado = 200
alto_deseado = 200
def nothing(x):
    pass



def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min", nameWindow, 0, 255, nothing)
    cv2.createTrackbar("max", nameWindow, 202, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 81, 100, nothing)
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
                cv2.putText(imagenOriginal, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(imagenOriginal, text_aux, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(imagenOriginal, mensaje2["clase"], (300, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(imagenOriginal, mensaje1["clase"], (10, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(imagenOriginal, total_text, (310, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(imagenOriginal, procesar_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(imagenOriginal, procesar_text2, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(imagenOriginal, restar_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.drawContours(imagenOriginal, [figuraActual], 0, (0, 0, 255), 2)
                if start_counting:
                    squares_vertices.append(vertices.tolist())
                    if n_cartas == 2:
                        num_squares += 1
                        if num_squares == 2:
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
                            print('coordenadas ', x1, x2, x3, x4, x5, x6, x7, x8)
                            print('coordenadas ', y1, y2, y3, y4, y5, y6, y7, y8)
                            primer_cuadrado = imagenGris[min(y1, y2, y3, y4):max(y1, y2, y3, y4),
                                              min(x1, x2, x3, x4):max(x1, x2, x3, x4)]
                            segundo_cuadrado = imagenGris[min(y5, y6, y7, y8):max(y5, y6, y7, y8),
                                               min(x5, x6, x7, x8):max(x5, x6, x7, x8)]
                            primer_cuadrado = cv2.resize(primer_cuadrado, (ancho_deseado, alto_deseado))
                            segundo_cuadrado = cv2.resize(segundo_cuadrado, (ancho_deseado, alto_deseado))
                            if primer_cuadrado is not None and segundo_cuadrado is not None:
                                # Comparar las imágenes de los cuadrados utilizando SSIM
                                similarity = ssim(primer_cuadrado, segundo_cuadrado)
                                similarity_threshold = 0.4  # Ajusta el umbral de similitud según sea necesario
                                print(similarity)
                                if similarity > similarity_threshold:
                                    print("Los cuadrados son similares. No se guardarán las imágenes.")
                                    segundo_cuadrado = None
                                    squares_vertices = []
                                    num_squares = 0
                                else:

                                    # Guardar las imágenes de los cuadrados en archivos separados
                                    cv2.imwrite("imgs/dos/primer_carta.jpg", primer_cuadrado)
                                    cv2.imwrite("imgs/dos/segundo_carta.jpg", segundo_cuadrado)
                                    start_counting = False
                                    squares_vertices = []
                                    num_squares = 0
                            else:
                                squares_vertices = []
                                num_squares = 0
                    elif n_cartas == 1:
                        vertices1 = squares_vertices[0]  # Coordenadas del primer cuadro
                        x1 = vertices1[0][0][0]
                        y1 = vertices1[0][0][1]

                        x2 = vertices1[1][0][0]
                        y2 = vertices1[1][0][1]

                        x3 = vertices1[2][0][0]
                        y3 = vertices1[2][0][1]

                        x4 = vertices1[3][0][0]
                        y4 = vertices1[3][0][1]
                        carta = imagenGris[min(y1, y2, y3, y4):max(y1, y2, y3, y4),
                                min(x1, x2, x3, x4):max(x1, x2, x3, x4)]
                        cv2.imwrite("imgs/una/carta.jpg", carta)
                        squares_vertices = []
                        start_counting = False
                else:
                    continue
        i += 1
    return imagenOriginal

video = cv2.VideoCapture(1)
constructorVentana()

while True:
    _, frame = video.read()
    k = cv2.waitKey(5)
    if capture_photo and start_counting is False:
        cv2.imshow("Video", frame)
    else:
        frame = detectarFigura(frame)
        if show_opciones:
             procesar_text = 'Pulse 1 para predecir individualmente'
             procesar_text2 = 'Pulse 2 para predecir dos cartas'
             mode_text = "Elija opción de predicción"
        else:
             procesar_text = 'Pulse C para tomar la foto'
             procesar_text2 = ''
             total_text = f'Total {total}'
        cv2.namedWindow("Imagen", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Imagen", 600, 750)
        cv2.imshow("Imagen", frame)
    if k == ord('1'):
        show_opciones = False
        n_cartas = 1
        mode_text = "Predecir una carta"
        procesar_text2 = ''
    elif k == ord('2'):
        show_opciones = False
        procesar_text2 = ''
        mode_text = "Predecir dos cartas"
        n_cartas = 2
    elif k == ord('q') or k == ord('Q'):  # Press 'q' to exit the program
        break
    if show_opciones is False:
        if k == ord('c') or k == ord('C'): # Press 'c' to capture a photo
            procesar_text = 'Pulse P para procesar la foto'
            restar_text = 'Pulse R para tomar otra foto'
            capture_photo = True
            start_counting = True
        elif k == ord('p') or k == ord('P'):
            if prediccion is False:
                text_aux = "Ultima(s) carta(s):"
                restar_text = ""
               # pruebas()
                mensaje = pruebasCNN.process(n_cartas)
                if len(mensaje) == 2:
                   mensaje1 = mensaje[0]
                   mensaje2 = mensaje[1]
                   total += mensaje1["valor"]
                   total += mensaje2["valor"]
                else:
                   mensaje1 = mensaje[0]
                   mensaje2 = {'clase': ''}
                   total+= mensaje1["valor"]
                prediccion = True
                print(f'se detectó {mensaje}')
                capture_photo = False
    else:
        mode_text = "Debe seleccionar un modo"
    if k == ord('r') or k == ord('R'):  # Press 'r' to restart counting
        show_opciones = True
        restar_text = ""
        capture_photo = False
        prediccion = False
        cv2.destroyWindow("Video")


video.release()
cv2.destroyAllWindows()
