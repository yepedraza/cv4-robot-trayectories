import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model_1.h5')
image = cv2.imread('ProveImage2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
No_Noise = cv2.fastNlMeansDenoising(gray, None,25,7,21)
_, th = cv2.threshold(No_Noise, 170, 255, cv2.THRESH_BINARY) #Modificar a conveniencia
border = cv2.Canny(th, 170, 200) #Modificar a conveniencia

contours, hierarchy = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
idGray = gray.copy()

for cont in contours:
    epsilon = 0.01*cv2.arcLength(cont, True) #Modificar a conveniencia
    aprox =  cv2.approxPolyDP(cont, epsilon, True)
    aprox = aprox.reshape(aprox.shape[0], aprox.shape[-1])

    cv2.rectangle(idGray, (aprox[1, 0], aprox[1, 1]), (aprox[-1, 0], aprox[-1, 1]), (0,0,255), 2)

    # cv2.circle(idGray, (aprox[1, 0], aprox[1, 1]), 30, (0,0,255) , 5)
    # cv2.circle(idGray, (aprox[-1, 0], aprox[-1, 1]), 30, (0,0,255) , 5)

    # print('\ndim:', idGray.shape)
    # print(type(idGray))
    # print('x:', aprox[1, 0], aprox[-1, 0],'\ny:', aprox[1, 1], aprox[-1, 1])

    content = idGray[ aprox[1, 1]:aprox[-1, 1], aprox[1, 0]:aprox[-1, 0] ]
    content = cv2.resize(content, (28, 28), interpolation = cv2.INTER_CUBIC) #Tamaño de la imágen re hecha

    # print(prediction)
    # POSIBLES ARREGLOS RED NEURONAL
    _, th2 = cv2.threshold(content, 170, 255, cv2.THRESH_BINARY)
    # print(th2)
    th2 = 255 - th2

    toPredictContent = (th2.reshape((1, 28, 28, 1))).astype('float32') / 255.0
    # prediction = model.predict(toPredictContent)
    prediction = np.argmax(model.predict(toPredictContent)[0], axis=-1)
    print("Prediction: ", prediction)

    cv2.imshow('Imagen', th2)
    cv2.waitKey(0)

# cv2.imshow('Imagen', No_Noise)
# cv2.imshow('Imagen', gray)
cv2.imshow('border', th)
cv2.waitKey(0)
cv2.destroyAllWindows()
