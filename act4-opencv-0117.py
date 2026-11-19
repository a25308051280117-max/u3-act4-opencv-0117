
# ==============================================================================
# 0. IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import cv2  # Importa la librería OpenCV para procesamiento de imágenes y visión artificial.
import numpy as np  # Importa NumPy para manejar las imágenes como arreglos/matrices matemáticas.
import sys  # Importa Sys para poder detener el programa de forma segura si ocurre un error.

# ==============================================================================
# 1. LECTURA DE LA IMAGEN DESDE LA CARPETA
# ==============================================================================
print("\n1. LECTURA DE IMÁGENES")  # Imprime el título de la sección en la consola.
imagencarrorojo = cv2.imread('carrorojo.jpeg')  # Lee la imagen 'carrorojo.jpeg' y la guarda como matriz en 'img'.
print("-+-+-+-+-Matias Lopez NC 0117-+-+-+-+-+-")

if imagencarrorojo is None:
    print("Error: No se pudo leer 'carrorojo.jpeg'. Revisa el nombre y ubicación.")
    sys.exit()
else:
    print("Imagen cargada exitosamente.")

# Agregar nombre y número de control a la imagen original
cv2.putText(imagencarrorojo, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 2. PROPIEDADES DE LA IMAGEN
# ==============================================================================
print("\n2. PROPIEDADES DE LA IMAGEN")
height, width, channels = imagencarrorojo.shape
print(f"Dimensiones de la imagen: {width}x{height} píxeles")
print(f"Número de canales de color: {channels}")
print(f"Tipo de dato de la imagen: {imagencarrorojo.dtype}")
print(f"Total de datos/píxeles: {imagencarrorojo.size}")

# ==============================================================================
# 3. VISUALIZACIÓN DE LA IMAGEN ORIGINAL
# ==============================================================================
print("\n3. MOSTRANDO IMAGEN ORIGINAL")
print("Presiona cualquier tecla sobre la ventana de la imagen para continuar...")
cv2.imshow('imagencarrorojo NC 0117 matias lopez', imagencarrorojo)
cv2.waitKey(0)

# ==============================================================================
# 4. ACCESO Y MODIFICACIÓN DE PÍXELES INDIVIDUALES
# ==============================================================================
print("\n4. ACCEDIENDO Y MODIFICANDO PÍXELES")
pixel_img = imagencarrorojo.copy()

y, x = 100, 150
pixel = pixel_img[y, x]
print(f"Valor del píxel en ({x}, {y}): {pixel} (Formato BGR)")

pixel_img[y, x] = [0, 0, 255]
cv2.circle(pixel_img, (x, y), 15, (0, 255, 255), 2)

cv2.putText(pixel_img, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Pixel Modificado NC 0117 Matias Lopez', pixel_img)
cv2.waitKey(0)

# ==============================================================================
# 5. REGIÓN DE INTERÉS (ROI - REGION OF INTEREST)
# ==============================================================================
print("\n5. REGIÓN DE INTERÉS (ROI)")
roi_img = imagencarrorojo.copy()

roi = roi_img[50:200, 100:300]
cv2.rectangle(roi_img, (100, 50), (300, 200), (0, 255, 0), 2)

cv2.putText(roi_img, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Imagen con ROI marcada', roi_img)
cv2.waitKey(0)

roi = roi.copy()
cv2.putText(roi, 'MATIAS LOPEZ NC 0117', (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imshow('ROI Extraida (Recorte)', roi)
cv2.waitKey(0)

# ==============================================================================
# 6. SEPARACIÓN Y FUSIÓN DE CANALES DE COLOR (BGR)
# ==============================================================================
print("\n6. SEPARANDO Y FUSIONANDO CANALES")
b, g, r = cv2.split(imagencarrorojo)

cv2.putText(b, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
cv2.imshow('Canal Azul (Escala de grises)', b)

cv2.putText(g, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
cv2.imshow('Canal Verde (Escala de grises)', g)

cv2.putText(r, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
cv2.imshow('Canal Rojo (Escala de grises)', r)

cv2.waitKey(0)

zeros = np.zeros_like(b)
blue_img = cv2.merge([b, zeros, zeros])
green_img = cv2.merge([zeros, g, zeros])
red_img = cv2.merge([zeros, zeros, r])

cv2.putText(blue_img, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(green_img, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(red_img, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Canal Azul (Coloreado)', blue_img)
cv2.imshow('Canal Verde (Coloreado)', green_img)
cv2.imshow('Canal Rojo (Coloreado)', red_img)
cv2.waitKey(0)

# ==============================================================================
# 7. REDIMENSIONAMIENTO DE IMÁGENES (RESIZING)
# ==============================================================================
print("\n7. REDIMENSIONANDO IMÁGENES")
resized_img = cv2.resize(imagencarrorojo, (400, 300))
cv2.putText(resized_img, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow('Redimensionada a 400x300', resized_img)
cv2.waitKey(0)

half_size = cv2.resize(imagencarrorojo, None, fx=0.5, fy=0.5)
double_size = cv2.resize(imagencarrorojo, None, fx=1.5, fy=1.5)

cv2.putText(half_size, 'MATIAS LOPEZ NC 0117', (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
cv2.putText(double_size, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Mitad de tamano (0.5x)', half_size)
cv2.waitKey(0)
cv2.imshow('Ampliada (1.5x)', double_size)
cv2.waitKey(0)

# ==============================================================================
# 8. ROTACIÓN DE IMÁGENES
# ==============================================================================
print("\n8. ROTANDO IMÁGENES")
height, width = imagencarrorojo.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
rotated_img = cv2.warpAffine(imagencarrorojo, rotation_matrix, (width, height))

cv2.putText(rotated_img, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Imagen Rotada (45 grados)', rotated_img)
cv2.waitKey(0)

# ==============================================================================
# 9. VOLTEO / ESPEJO DE IMÁGENES (FLIPPING)
# ==============================================================================
print("\n9. VOLTEANDO IMÁGENES")
horizontal_flip = cv2.flip(imagencarrorojo, 1)

cv2.putText(horizontal_flip, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Volteo Horizontal (Espejo)', horizontal_flip)
cv2.waitKey(0)

vertical_flip = cv2.flip(imagencarrorojo, 0)

cv2.putText(vertical_flip, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Volteo Vertical (De cabeza)', vertical_flip)
cv2.waitKey(0)

both_flip = cv2.flip(imagencarrorojo, -1)

cv2.putText(both_flip, 'MATIAS LOPEZ NC 0117', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Volteo en Ambos Ejes', both_flip)
cv2.waitKey(0)

# ==============================================================================
# 10. GUARDADO DE RESULTADOS Y LIMPIEZA
# ==============================================================================
print("\n10. GUARDANDO IMAGEN RESULTANTE")
result = cv2.imwrite('rotated_image_0117.jpg', rotated_img)

if result:
    print("Imagen rotada guardada exitosamente como 'rotated_image_0117.jpg'")
else:
    print("Error: No se pudo guardar la imagen rotada.")

cv2.destroyAllWindows()
print("\n¡Todas las operaciones se completaron con éxito!")
print("Matias Lopez NC 0117")
