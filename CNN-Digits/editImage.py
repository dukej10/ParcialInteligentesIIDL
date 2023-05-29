import os
import shutil

import cv2

# Métodos para editar el dataset

def convertir_grises_imagenes():
    # Obtener la ruta del archivo de script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta completa de la imagen
    for n in range(6,13):
        for i in range(0,42):
            image_path = os.path.join(script_dir, "dataset", "test", str(n), str(n)+'_' + str(i) + '.jpg')
            # Cargar la imagen
            img = cv2.imread(image_path)

            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))

            # Guardar en test
            output_folder = os.path.join(script_dir, "dataset", "test","grays", str(n))
            os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe
            output_path = os.path.join(script_dir, "dataset", "test","grays", str(n), str(n)+'_' + str(i) + '.jpg')
            cv2.imwrite(output_path, gray)
            #Guardar en train
            output_folder = os.path.join(script_dir, "dataset", "train", "grays", str(n))
            os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe
            output_path = os.path.join(script_dir, "dataset", "train", "grays", str(n), str(n) + '_' + str(i) + '.jpg')
            cv2.imwrite(output_path, gray)

def resize():
    # Obtener la ruta del archivo de script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for n in range(6,13):
        for i in range(0,42):
            image_path = os.path.join(script_dir, "dataset", "test", str(n), str(n)+'_' + str(i) + '.jpg')            # Cargar la imagen
            img = cv2.imread(image_path)

            # Escalar imagenes
            img = cv2.resize(img, (128, 128))
            #Guardar imagenes en test
            output_folder = os.path.join(script_dir, "dataset", "test", "grays", str(n))
            os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe
            output_path = os.path.join(script_dir, "dataset", "test", "grays",str(n), str(n)+'_' + str(i) + '.jpg')
            cv2.imwrite(output_path, img)
            # Guardar imagenes en train
            output_folder = os.path.join(script_dir, "dataset", "train", "grays", str(n))
            os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe
            output_path = os.path.join(script_dir, "dataset", "train", "grays", str(n), str(n) + '_' + str(i) + '.jpg')
            cv2.imwrite(output_path, img)

def rotar_imagenes():
    # Obtener la ruta del archivo de script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    num = 0
    for n in range(6, 13):
        nimg = 0
        num = 0
        for i in range(0, 42):
            if num > 0:
                num = num -1
            image_path = os.path.join(script_dir, "dataset", "test", "grays", str(n), str(n) + '_' + str(i) + '.jpg')
            print(image_path)
            # Cargar la imagen
            img = cv2.imread(image_path)
            # Rotar imagen cada 72 grados
            for rotation in range(72, 360, 72):
                # Obtener dimensiones de la imagen
                rows, cols, _ = img.shape
                # crear matriz para indicar como rotar
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
                #rotar la imagen
                rotated_img = cv2.warpAffine(img, M, (cols, rows))
                # Guardar imagen rotada
                # Guardar en test
                output_folder = os.path.join(script_dir, "dataset", "test", "grays", str(n))
                os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe
                output_path = os.path.join(script_dir, "dataset", "test", "grays", str(n), str(n) + '_' + str(i+42+num-1) + '.jpg')
                cv2.imwrite(output_path, rotated_img)
                # Guardar en train
                output_folder = os.path.join(script_dir, "dataset", "train", "grays", str(n))
                os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe
                output_path = os.path.join(script_dir, "dataset", "train", "grays", str(n),
                                           str(n) + '_' + str(i + 42 + num - 1) + '.jpg')
                cv2.imwrite(output_path, rotated_img)
                print(output_path)
                #print(output_path)

                num = num + 1

            nimg = nimg + 1
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Eliminar el contenido de la carpeta
    output_folder = os.path.join(script_dir, "dataset", "test", "grays")
    shutil.rmtree(output_folder)
    # Eliminar el contenido de la carpeta
    output_folder = os.path.join(script_dir, "dataset", "train", "grays")
    shutil.rmtree(output_folder)
    # Invocar métodos encargados de la transformación de las imágenes
    resize()
    convertir_grises_imagenes()
    rotar_imagenes()


main()