from ultralytics import YOLO
import cv2

# Cargar modelo YOLO preentrenado (YOLOv8n: rápido y ligero)
modelo = YOLO('yolov8n.pt')

# Inicializar cámara (0 = cámara por defecto)
camara = cv2.VideoCapture(0)

while True:
    ret, frame = camara.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break


    # Realizar la detección
    resultados = modelo(frame)[0]

    # Mostrar los resultados con anotaciones
    frame_etiquetado = resultados.plot()
    cv2.imshow("Detección de Objetos (YOLOv8)", frame_etiquetado)

    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()
