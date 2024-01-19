import tensorflow as tf
import cv2
import numpy as np
import time

# Cargar el modelo TensorFlow Lite
#model.tflite#Modelo Segmentación 640x640
#modelo entrenado 96x96 segmentacion #modelos/best_float32_96.tflite
interpreter = tf.lite.Interpreter(model_path='modelos/yolov8-seg-linea_96.tflite')
#interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Abrir el archivo de video
video_path = 'test/video8.mp4'  # Cambia 'video.mp4' por la ruta de tu video
output_path = 'outputVideo/output_video8_tflite.mp4'  # Nombre y ruta del video de salida

precision = 0.1

# Obtener los detalles de entrada y salida del modelo tflite
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
input_size=input_shape[2]
print("La dimensiones de entrada del modelo son:",input_shape) 

output_details = interpreter.get_output_details()

print("Numero de Salidas:",len(output_details))

output_shape0 = output_details[0]['shape']
print("La dimensiones de salida 0 del modelo son:",output_shape0)

if(len(output_details)>1):
    output_shape1 = output_details[1]['shape']
    print("La dimensiones de salida 1 del modelo son:",output_shape1)
    dimension1 = output_shape1[2]
    dimension2 = output_shape1[3] #32

#print("input_details:",input_details) #[{'name': 'serving_default_images:0', 'index': 0, 'shape': array([  1,   3, 640, 640]), 'shape_signature': array([  1,   3, 640, 640]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
#print("input_shape:",input_shape)#[  1   3 640 640]    
#print("input_shape[2]:",input_shape[2])#[  1   3 640 640]

'''
La dimensiones de entrada del modelo son: [ 1 96 96  3]
La dimensiones de salida 0 del modelo son: [  1 116 189]
La dimensiones de salida 1 del modelo son: [ 1 24 24 32]
'''

dimension3 = output_shape0[1]#116

'''
La dimensiones de entrada del modelo son: [  1   3 640 640]
La dimensiones de salida 0 del modelo son: [   1  116 8400]
La dimensiones de salida 1 del modelo son: [  1  32 160 160]
'''


'''
# Imprimir los detalles de entrada del modelo
print("Detalles de entrada del modelo:")
for detail in input_details:
    print(detail)

# Imprimir los detalles de entrada del modelo
print("Detalles de salida del modelo:")
for detail in output_details:
    print(detail)'''


#video_capture = cv2.VideoCapture(0)

video_capture = cv2.VideoCapture(video_path)




# Obtener las dimensiones del video
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
print("FPS:",fps)
img_width = frame_width
img_height = frame_height

# Define las nuevas dimensiones deseadas

img_width = 640
img_height = 640

# Crear un objeto VideoWriter para guardar el video procesado

video_writer = cv2.VideoWriter(output_path,
                               cv2.VideoWriter_fourcc(*'mp4v'),  # Códec de video (puedes cambiarlo según el formato deseado)
                               fps, (img_width, img_height))

yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "linea", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_mask(row, box):
    mask = row.reshape(dimension1, dimension1)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype('uint8') * 255
    x1, y1, x2, y2 = box
    
    # Calculate mask coordinates
    mask_x1 = round(x1 * dimension1)
    mask_y1 = round(y1 * dimension1)
    mask_x2 = round(x2 * dimension1)
    mask_y2 = round(y2 * dimension1)
    
    if round(y2) == 0:
        y2 = y2 + 2

    if round(x2) == 0:
       x2 = x2 + 2
    
    if mask_y2 == 0:
        mask_y2 = round(y2)

    if mask_x2 == 0:
        mask_x2 = round(x2)

    # Check if the calculated dimensions are valid before resizing
    if mask_x1 < mask_x2 and mask_y1 < mask_y2:
        mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]

        x1 = x1 * img_width
        y1 = y1 * img_height
        x2 = x2 * img_width
        y2 = y2 * img_height

        # Check if the calculated dimensions are valid before resizing
        if x2 - x1 > 0 and y2 - y1 > 0:
            if x2 > x1 and y2 > y1:
                if not mask.size == 0:  # Check that the mask is not empty
                    # Resize the mask
                    mask = cv2.resize(mask, (round(x2 - x1), round(y2 - y1)))
    else:
        mask = np.zeros((0, 0), dtype=np.uint8)  # Create an empty image

    return mask

def get_polygon(mask):
    contours = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #polygon = [[contour[0][0],contour[0][1]] for contour in contours[0][0]]
    # Check if there are any contours before accessing them
    if len(contours) > 0 and len(contours[0]) > 0:
        polygon = [[contour[0][0], contour[0][1]] for contour in contours[0][0]]
    else:
        # Handle the case where there are no contours
        polygon = []  # or any other appropriate action
    return polygon

# Bucle para procesar cada fotograma
# Bucle para procesar cada fotograma
while True:
    #start_capture_time = time.time()
    # Medir el tiempo de inicio del procesamiento del cuadro
    start_processing_time  = time.time()

    ret, frame = video_capture.read()

    # Medir el tiempo de finalización de captura del cuadro
    #end_capture_time = time.time()


    if not ret:
        break

    
    frame_resized = cv2.resize(frame, (img_width, img_height))

    # Redimensionar el frame a 320x320 o 640x640(si es necesario)
    frame_resized2 = cv2.resize(frame, (input_size, input_size))

    # Convertir el frame a formato RGB (OpenCV utiliza BGR por defecto)
    frame_rgb = cv2.cvtColor(frame_resized2, cv2.COLOR_BGR2RGB)

    # Convertir el frame a un tensor [1, 3, input_size, input_size] requerido para el modelo
    frame_input = np.array(frame_rgb)
    frame_input = frame_input.transpose(2, 0, 1)
    frame_input = frame_input.reshape(input_shape).astype('float32')
    frame_input = frame_input / 255.0

    # Obtener las salidas del modelo tflite
    interpreter.set_tensor(input_details[0]['index'], frame_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
       
    output0 = output_data
    
    #print("outputs:",outputs.shape)#(1, 116, 8400)#(1, 32, 160, 160)

    
    # Obtener las salidas del modelo onnx
    #outputs = model.run(None, {"images": frame_input})
    #output0 = outputs[0]
    #output1 = outputs[1]
    #print("output0:",output0.shape)#(1, 116, 2100)
    #print("output1:",output1.shape)#(1, 32, 80, 80)
    
    
    output0 = output0[0].transpose() #[189,116]
    
    #output1.shape[2]
    # Resto del código de procesamiento (boxes, masks, etc.)

    # El resto del código permanece igual...

    boxes = output0[:,0:84]
    
    #print(masks.shape) #(8400, 32)

    
    if(len(output_details)>1):
        masks = output0[:,84:]

        output_data1 = interpreter.get_tensor(output_details[1]['index'])
        output1 = output_data1
        output1 = output1[0]

        #Redimencionar:
        if(dimension2 == 32):
            output1 = np.transpose(output1, (2, 0, 1))
    
        #print(output1.shape)
        output1 = output1.reshape(32,dimension1*dimension1)

        #Multiplicación matricial
        masks = masks @ output1
        #print(masks.shape) #(8400, 25600)

        #Ahora los conectaremos juntos. Agreguemos 25600 columnas de la segunda matriz a la primera:
        boxes = np.hstack((boxes,masks))




    objects = []
    for row in boxes:
        prob = row[4:84].max()
        if prob < precision:
            continue        
        xc,yc,w,h = row[:4]
        class_id = row[4:84].argmax()
        
        x1 = (xc-w/2)/input_size
        y1 = (yc-h/2)/input_size
        x2 = (xc+w/2)/input_size
        y2 = (yc+h/2)/input_size

        if(len(output_details)>1):
            if(dimension2 == 32):
                x1 = (xc-w/2)
                y1 = (yc-h/2)
                x2 = (xc+w/2)
                y2 = (yc+h/2)       

        label = yolo_classes[class_id]
        #print(row.shape)

        
        if(len(output_details)>1):
            mask = get_mask(row[84:(84+(dimension1*dimension1))],(x1,y1,x2,y2))#((84+(dimension1*dimension1))+1)
            #mask = get_mask(row[84:dimension3],(x1,y1,x2,y2))#((84+(dimension1*dimension1))+1)
            polygon = get_polygon(mask)
            objects.append([x1,y1,x2,y2,label,prob,polygon])
        else:
            objects.append([x1,y1,x2,y2,label,prob])
        


    #for obj in objects:
    #    print(obj[5])  # Imprime el valor de x[5]
    objects.sort(key=lambda x: x[5], reverse=True)
    #print("cantidad:",len(objects))
    result = []
    while len(objects)>0:
        result.append(objects[0])
        objects = [object for object in objects if iou(object,objects[0])<0.7]

    #print("cantidad:",len(result))

    for object in result:

        if(len(output_details)>1):        
            [x1, y1, x2, y2, label, prob, polygon] = object
            
            if label == "linea":
                if prob > 0.5:
                    # Ensure that the polygon points stay within the rectangle
                    polygon = [(int((x1*img_width) + point[0]), int((y1*img_height) + point[1])) for point in polygon]
                    for i in range(len(polygon)):
                        polygon[i] = (max(min(polygon[i][0], int(x2*img_width)), int(x1*img_width)),
                                    max(min(polygon[i][1], int(y2*img_height)), int(y1*img_height)))

                    #polygon = [(int((x1*img_width) + point[0]), int((y1*img_width)  + point[1])) for point in polygon]
                    # Dibujar un polígono con transparencia en OpenCV
                    #cv2.fillPoly(frame, [np.array(polygon)], (0, 255, 0, 125))
                    # Dibujar un polígono con trazo delgado en OpenCV (thickness = 1)
                    #cv2.polylines(frame_resized, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Calcular el punto medio entre el primer y último vértice del polígono
                    mid_point_x = (polygon[0][0] + polygon[-1][0]) // 2
                    mid_point_y = (polygon[0][1] + polygon[-1][1]) // 2
                    mid_point = (mid_point_x, mid_point_y)

                    # Dibujar una línea recta que conecte el punto medio con el centro del rectángulo
                    center_x = (x1 + x2)*img_width // 2
                    center_y = (y1 + y2)*img_height // 2
                    mid_point = (int(mid_point_x), int(mid_point_y))                    
                    #mid_point = (int(x2*img_width), int(y1*img_width))
                    center_point = (int(center_x), int(center_y))
                    cv2.line(frame_resized, mid_point, center_point, (0, 0, 255), thickness=2)
                    cv2.rectangle(frame_resized, (int(x1*img_width), int(y1*img_height)), (int(x2*img_width), int(y2*img_height)), (255, 0, 0), 1)
                    x_text = int(x1*img_width)  # Puedes ajustar estas coordenadas según tus necesidades
                    y_text = int(y1*img_height) + 10  # Desplaza el texto ligeramente hacia arriba para que no se superponga con el rectángulo

                    # Coloca el letrero dentro del rectángulo
                    cv2.putText(frame_resized, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                        
                    cv2.putText(frame_resized, str(round(prob*100))+"%", (x_text, y_text+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            else:
                    cv2.rectangle(frame_resized, (int(x1*img_width), int(y1*img_height)), (int(x2*img_width), int(y2*img_height)), (255, 120, 190), 2)
                    x_text = int(x1*img_width)  # Puedes ajustar estas coordenadas según tus necesidades
                    y_text = int(y1*img_height) + 10  # Desplaza el texto ligeramente hacia arriba para que no se superponga con el rectángulo

                    # Coloca el letrero dentro del rectángulo
                    cv2.putText(frame_resized, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                        
                    cv2.putText(frame_resized, str(round(prob*100))+"%", (x_text, y_text+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                        
            
                
        
        else:
            [x1, y1, x2, y2, label, prob] = object
            cv2.rectangle(frame_resized, (int(x1*img_width), int(y1*img_height)), (int(x2*img_width), int(y2*img_height)), (255, 0, 0), 2)
            x_text = int(x1*img_width)  # Puedes ajustar estas coordenadas según tus necesidades
            y_text = int(y1*img_height) + 10  # Desplaza el texto ligeramente hacia arriba para que no se superponga con el rectángulo

            # Coloca el letrero dentro del rectángulo
            cv2.putText(frame_resized, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                
            cv2.putText(frame_resized, str(round(prob*100))+"%", (x_text, y_text+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
        

        
        
        '''
                

            
        else:
            # Dibujar un rectángulo con borde en OpenCV
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        '''
    

    end_processing_time  = time.time()
    # Calcular el tiempo de procesamiento del cuadro
    processing_time = end_processing_time  - start_processing_time 
    # Calcular el tiempo de latencia (diferencia entre el tiempo de finalización de captura y el tiempo de finalización de procesamiento)
    #latency = end_processing_time - end_capture_time
    #print("Tiempo de ejecución ms",round(processing_time*1000))
    # Mostrar el frame procesado con información del tiempo de procesamiento
    cv2.putText(frame_resized, f"Tiempo de Procesamiento: {round(processing_time*1000):d} mseg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 255, 220), 2)
    #cv2.putText(frame, f"Latencia: {latency:.5f} seg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    # Mostrar el frame procesado
    #cv2.imshow("Resultado", frame_resized)

    # Medir el tiempo de finalización del procesamiento del cuadro
    

    
    tiempo_e = (1/fps) - processing_time

    if (tiempo_e > 0):
        time.sleep(tiempo_e)
    #else:
        #print("Tiempo perdido",(-1*tiempo_e))
    
    # Guardar el frame procesado en el video de salida
    video_writer.write(frame_resized)
   
    # Esperar una tecla para detener la visualización (puedes eliminar estas líneas si no quieres visualizar el video)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Liberar los objetos de video
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
