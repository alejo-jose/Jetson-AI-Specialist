# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 20:04:19 2023

@author: Alejandro
"""

import torch

import cv2
import numpy as np
#from PIL import Image 


from keras.models import load_model  # TensorFlow is required for Keras to work
#from PIL import Image, ImageOps  # Install pillow instead of PIL
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



### Function to load models and process the image ####

def procesar_imagen():
   # Load AI models #
    ruta_yolo = '/home/jetson/Documentos/prueba/modelo/best.pt' #Location of AI model 1
    ruta_etiquetas = '/home/jetson/Documentos/prueba/modelo_cnn/labels.txt' #Location of labels for AI model 2
    
    model = torch.hub.load('ultralytics/yolov5', 'custom',path = ruta_yolo) #Load AI model 1
    #print("modelo yolo")

    modelo = load_model('/home/jetson/Documentos/prueba/modelo_cnn/keras_model_new.h5') #Load AI model 2
    #print("modelo cnn")

    with open(ruta_etiquetas, 'r') as f:
        etiquetas = f.read().split('\n')
    print("etiquetas")
    
    u=0
    n=0
    naa=0
    
    #Create a list to store the central points of each figure
    puntos_centrales = []
    total=[]
    cuadrante=[]
    
    for i in range(9,10,1):
        x="/home/jetson/Documentos/prueba/imagenes3/foto_"+str(i)+".jpg"   #Location of the photo to be analyzed
        print(x)
        imagen = cv2.imread(x)

        #Quadrants
    
        ancho = imagen.shape[1] #Width
        alto = imagen.shape[0] #Heigh
        print(ancho)
        print(alto)
        
        #Matrix with 5 columns and 3 rows

        ancho_cuadrante = int(ancho/5)
        alto_cuadrante = int(alto/3)
        print("ancho_cuadrante: ",ancho_cuadrante)
        print("alto_cuadrante: ",alto_cuadrante)

        #End quadrants    
        
        #Starts AI
    
        detect = model(imagen) #AI model 1
    
        info = detect.pandas().xyxy[0]  #predictions
        print(info)
        nn="resultado"+str(n)+".jpg"
        print(nn)
        nnn="/home/jetson/Documentos/nvidia/imagenes/yolo/"+nn
        nu=np.squeeze(detect.render())        
        cv2.imwrite(nnn,nu)   #Save image with detections from AI model 1        
        n=n+1
        nu=np.squeeze(detect.render())  

        # Iterating over the detections
   
        for j in range(len(info)):

            # Obtain the coordinates of the region of interest
            x1, y1, x2, y2 = info.loc[j, ['xmin', 'ymin', 'xmax', 'ymax']]
    
            #  Region of interest
            region = imagen[int(y1):int(y2), int(x1):int(x2)]
            centro_x = int((x1 + x2) / 2)
            centro_y = int((y1 + y2) / 2)
            
            # Central quadrant of the matrix
            puntos_centrales.append((centro_x, centro_y))
            # End of center quadrant matrix
    
            nf="resultado_pieza"+str(u)+".jpg"
            print(nf)  
            nnf='/home/jetson/Documentos/nvidia/imagenes/individual/'+nf      
            cv2.imwrite(nnf,region) #Save individual image of the matrix          
            u=u+1

            # Analyze each space of the matrix
            ruta_imagen2 = nnf      
            image = cv2.imread(ruta_imagen2)

            image = cv2.resize(image, (224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
    
            # Perform the prediction with the AI model 2
            resultado = modelo.predict(image)
            indice_clase = np.argmax(resultado)
            etiqueta_predicha = etiquetas[indice_clase]
            print('La etiqueta predicha es:', etiqueta_predicha)
            # Display the prediction
            print(resultado)
            cv2.putText(imagen, etiqueta_predicha, (centro_x, centro_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)       
            #Finish iterating over detections

        #Verify information
        print(puntos_centrales) 
        print("ordenado 2")
        coordenadas_ordenadas = sorted(puntos_centrales, key=lambda x: (x[0], x[1]))
        print(coordenadas_ordenadas)
        
        flag_num=1
        yu=0
        xu=0
        i=1
        ## Segment the image ##
        for y in range(0, alto, alto_cuadrante):
            print("y ",y)
            yu=yu+1
            
            for x in range(0, ancho, ancho_cuadrante):
                print("x",x)
                xu=xu+1
                
                # Retrieve the upper-left corner and the lower-right corner of each quadrant
                print(coordenadas_ordenadas[flag_num][0])
                print(coordenadas_ordenadas[flag_num][1])
                
                x1 = x
                y1 = y
                x2 = x + ancho_cuadrante
                y2 = y + alto_cuadrante
    
                # Draw a rectangle on the image
                cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
                centro_x = (x1 + x2) // 2
                centro_y = (y1 + y2) // 2
                centro = (centro_x, centro_y+20)
                cv2.putText(imagen, str(i), centro, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
                print(f"Cuadrante ({x1}, {y1}) - ({x2}, {y2})") 
    
                # Perform the search in each quadrant
                cuadrante.append([x1,x2,y1,y2])
                # Iterate through the list of central points
                i=i+1
    
        ## Verify and store AI results ##
        print("nueva iteraccion")        
        flagg=False
        print(len(cuadrante))
        print(len(coordenadas_ordenadas))
        for ko in range(len(cuadrante)): 
          print("valor bandra ", flagg)
          flagg=False
          for h in range(len(coordenadas_ordenadas)):
            if cuadrante[ko][0] <= coordenadas_ordenadas[h][0] <= cuadrante[ko][1] and cuadrante[ko][2] <= coordenadas_ordenadas[h][1] <= cuadrante[ko][3]:
              # The point is located in the quadrant
              print(f"El punto ({coordenadas_ordenadas[h][0]}, {coordenadas_ordenadas[h][1]}) se encuentra en el cuadrante {ko}")
              flagg = True
              break
            else:
              #Not found
              print("vacio")

          #Store information in vector 'total
          if flagg:
              total.append([flag_num, "pieza"])
          else:
              total.append([flag_num, "disponible"])
    
          
          print(flag_num)
          print(" ")
            
          #print("valor x ", xu, "   valor y ",yu)
          print("orde x ",coordenadas_ordenadas[h][0])
          print("orde y ",coordenadas_ordenadas[h][1])
    
          flag_num =flag_num+1
          print("x1 ",cuadrante[ko][0]," x2 ",cuadrante[ko][1]," y1 ",cuadrante[ko][2]," y2 ",cuadrante[ko][3])
    
        na="resultado_original_color"+str(naa)+".jpg"
        print(na)  
        directorio_original='/home/jetson/Documentos/nvidia/imagenes/total/'+na 
        cv2.imwrite( directorio_original,imagen)   #Save segmented image with predictions
        naa=naa+1            
    print(total)


procesar_imagen()
