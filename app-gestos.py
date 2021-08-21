import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # leer imagen
    ret, img = cap.read()

    # obtener datos de la mano de la subventana del rectángulo en la pantalla
    cv2.rectangle(img, (400, 400), (100, 100), (0, 255, 0), 0)
    crop_img = img[100:400, 100:400]

    # convertir a escala de grises
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # aplicando desenfoque gaussiano
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # umbralización: método de binarización de Otsu
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # mostrar imagen de umbral
    cv2.imshow('Thresholded', thresh1)

    # verifique la versión de OpenCV para evitar errores de descompresión
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '4':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # encontrar contorno con área máxima
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # crear un rectángulo delimitador alrededor del contorno (puede saltar por debajo de dos líneas)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # encontrar casco convexo
    hull = cv2.convexHull(cnt)

    # dibujar contornos
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # encontrar casco convexo
    hull = cv2.convexHull(cnt, returnPoints=False)

    # encontrar defectos de convexidad
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # aplicar la regla del coseno para encontrar el ángulo de todos los defectos (entre los dedos)
    # con ángulo> 90 grados e ignorar defectos
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # hallar la longitud de todos los lados del triángulo
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # aplicar la regla del coseno aquí
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignorar ángulos> 90 y resaltar el resto con puntos rojos
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
        cv2.line(crop_img,start, end, [0,255,0], 2)
        #cv2.circle(crop_img,far,5,[0,0,255],-1)

    # define actions required
    if count_defects == 0:
        cv2.putText(img,"Detecto 1 dedo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 1:
        str = "Detecto 2 dedos"
        cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 2:
        cv2.putText(img,"Detecto 3 dedos.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(img,"Detecto 4 dedos", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(img,"Detecto 5 dedos", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    # mostrar imágenes apropiadas en Windows
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break
