import cv2
import numpy as np
import os
from scipy.spatial import distance
from PruebasCNN import PruebasCNN
pruebasCNN = PruebasCNN()
nameWindow = "Calculadora"
prediccion = False
string = ""
mensaje = {"clase": ""}
procesar_text = ""

# Variable to keep track of whether two squares have been detected
num_squares = 0

# Variable to control capturing the photo
capture_photo = False

# Variable to control counting squares
start_counting = False

# List to store the processed square vertices
processed_squares = []

# List to store the processed square centers
processed_centers = []

# List to store the detected square vertices
squares_vertices = []

total = 0

def nothing(x):
    pass

def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min", nameWindow, 0, 255, nothing)
    cv2.createTrackbar("max", nameWindow, 100, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 1, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 500, 10000, nothing)

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
    #cv2.imshow("Bordes", bordes)
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    #cv2.imshow("Bordes Modificado", bordes)
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)
    i = 0
    list_vertices = []
    areaMin = cv2.getTrackbarPos("areaMin", nameWindow)
    for figuraActual in figuras:
        if areas[i] >= areaMin:
            vertices = cv2.approxPolyDP(figuraActual, 0.05 * cv2.arcLength(figuraActual, True), True)
            if len(vertices) == 4:  # Verificar si la figura es un cuadrado y convexa:
                cv2.putText(imagenOriginal, mensaje["clase"], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(imagenOriginal, f'Total {total}', (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(imagenOriginal, procesar_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.drawContours(imagenOriginal, [figuraActual], 0, (0, 0, 255), 2)
                if start_counting:
                        num_squares += 1
                        squares_vertices.append(vertices)
                        squares_vertices.append(vertices.tolist())
                        vertices1 = squares_vertices[0] # Coordenadas del primer cuadro
                        x1 = vertices1[0][0][0]
                        y1 = vertices1[0][0][1]

                        x2 = vertices1[1][0][0]
                        y2 = vertices1[1][0][1]

                        x3 = vertices1[2][0][0]
                        y3 = vertices1[2][0][1]

                        x4 = vertices1[3][0][0]
                        y4 = vertices1[3][0][1]
                        x_coords = [vertex[0][0] for vertex in vertices1]
                        y_coords = [vertex[0][1] for vertex in vertices1]
                        # print(x1, y1)
                        primer_cuadrado = imagenGris[min(y1, y2, y3, y4):max(y1, y2, y3, y4), min(x1, x2, x3, x4):max(x1, x2, x3, x4)]

                        print("------------------")

                       # print(vertices2[0][0][1])
                        #print(vertices[0][0][1])
                        print("-----------")
                        os.makedirs("imgs", exist_ok=True)
                        cv2.imwrite("imgs/primer_cuadrado.jpg", primer_cuadrado)
                        processed_squares.extend(squares_vertices)
                        num_squares = 0
                        squares_vertices = []
                        start_counting = False
                else:
                    continue

        i += 1
    return imagenOriginal

def pruebas():
    # Obtener la ruta del archivo de script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta completa de la imagen
    image_path = os.path.join(script_dir, "imgs", "primer_cuadrado.jpg")

    # Cargar la imagen
    img = cv2.imread(image_path)

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
        cv2.imwrite(f"contorno_{0}.jpg", cropped)


video = cv2.VideoCapture(1)
constructorVentana()

while True:
    _, frame = video.read()
    k = cv2.waitKey(5)
    if k == ord('q') or k == ord('Q') :  # Press 'q' to exit the program
        break
    elif k == ord('c') or k == ord('C'):
        procesar_text = 'Pulse E para procesar la foto'# Press 'c' to capture a photo
        capture_photo = True
        start_counting = False

    elif k == ord('e') or k == ord('E'):
        if prediccion is False:
            pruebas()
            mensaje = pruebasCNN.process()
            total+= mensaje["valor"]
            prediccion = True
            print(f'se detectó {mensaje}')
            capture_photo = False

    elif k == ord('r') or k == ord('R'):  # Press 'r' to restart counting
        capture_photo = False
        prediccion = False
        procesar_text = "Pulse C para tomar la foto"
    if capture_photo and start_counting is False:
        if prediccion is False:
            pruebas()
    else:
        frame = detectarFigura(frame)
        # Agregar texto a la imagen

        procesar_text = 'Pulse C para tomar la foto'
        cv2.imshow("Imagen", frame)

video.release()
cv2.destroyAllWindows()
