import cv2
import numpy as np
import tensorflow as tf

def mse(y, y_hat, d = False):
    if d:
        return y_hat-y
    else:
        return np.mean((y_hat - y)**2)

def relu(x, derivate = False):
    if derivate:
        x[x<=0] = 0
        x[x>0] = 1
        return x
    else:
        return np.maximum(0,x)

def sigmoid(x, derivate = False):
    if derivate:
        return np.exp(-x)/((np.exp(-x)+1)**2)
    else:
        return (1/(1+np.exp(-x)))

def Forward(params, x_data):

    params['A0'] = x_data

    params['Z1'] = (params['A0']@params['W1']) + params['b1'] 
    params['A1'] = relu( params['Z1']) 

    params['Z2'] = (params['A1']@params['W2']) + params['b2'] 
    params['A2'] = relu(params['Z2'])

    params['Z3'] = (params['A2']@params['W3']) + params['b3'] 
    params['A3'] = sigmoid(params['Z3'])

    output = params['A3']

    return params, output

model = np.load('../trainingModels/models_export/handmade_model.npy')
image = cv2.imread('ProveImage2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
No_Noise = cv2.fastNlMeansDenoising(gray, None,25,7,21)
_, th = cv2.threshold(No_Noise, 170, 255, cv2.THRESH_BINARY) #Modificar a conveniencia
border = cv2.Canny(th, 170, 200) #Modificar a conveniencia

contours, hierarchy = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
idGray = gray.copy()
points = []
points.append([0, np.array(image).shape[0]//2])
for cont in contours:
    epsilon = 0.01*cv2.arcLength(cont, True) #Modificar a conveniencia
    aprox =  cv2.approxPolyDP(cont, epsilon, True)
    aprox = aprox.reshape(aprox.shape[0], aprox.shape[-1])

    cv2.rectangle(idGray, (aprox[1, 0], aprox[1, 1]), (aprox[-1, 0], aprox[-1, 1]), (0,0,255), 2) #x, y


    # cv2.circle(idGray, (aprox[1, 0], aprox[1, 1]), 30, (0,0,255) , 5)
    # cv2.circle(idGray, (aprox[-1, 0], aprox[-1, 1]), 30, (0,0,255) , 5)

    # print('\ndim:', idGray.shape)
    # print(type(idGray))
    # print('x:', aprox[1, 0], aprox[-1, 0],'\ny:', aprox[1, 1], aprox[-1, 1])

    content = idGray[ aprox[1, 1]:aprox[-1, 1], aprox[1, 0]:aprox[-1, 0] ]
    content = cv2.resize(content, (28, 28), interpolation = cv2.INTER_CUBIC) #Tama??o de la im??gen re hecha

    # POSIBLES ARREGLOS RED NEURONAL
    _, th2 = cv2.threshold(content, 170, 255, cv2.THRESH_BINARY)
    # print(th2)
    th2 = 255 - th2

    toPredictContent = (th2.reshape((1, 28, 28, 1))).astype('float32') / 255.0
    toPredictContent = toPredictContent.reshape(1, 784)
    params, prediction_v = Forward(params, toPredictContent)
    prediction = np.argmax(prediction_v)
    print("\nPrediction: ", prediction)
    np.set_printoptions(precision = 3, suppress = True)
    print("Ownership Percentage:", prediction_v*100)#S, prediction_v*100)

    font = cv2.FONT_HERSHEY_SIMPLEX
    middle = (aprox[1, 0] + aprox[-1, 0])//2
    cv2.putText(idGray, 'prediction: {} {:.2f}%'.format(prediction, prediction_v[prediction]*100),
                (middle, aprox[-1, 1]+30), font, 0.8, (0,0,255), 2)

    points.append([aprox[1, 0], aprox[1, 1]])
    # print("Prediction: ", prediction)
    # cv2.imshow('Imagen', idGray)
    # cv2.waitKey(0)

points.append([np.array(image).shape[1], np.array(image).shape[0]//2])
points = np.array(points)


#### SE DIBUJA LA TRAYECTORIA, SIN EMBARGO NO SE HA ORDENADO ####
for i in range(points.shape[0]-1):
    cv2.line(idGray, (points[i,:]),(points[i+1,:]),(0,0,255),3)

# print('points:', points[0,:])
# cv2.imshow('Imagen', No_Noise)
# cv2.imshow('Imagen', gray)
cv2.imshow('border', idGray)
cv2.waitKey(0)
cv2.destroyAllWindows()