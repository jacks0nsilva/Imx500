import argparse
import sys
from functools import lru_cache


import numpy as np
import math

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

last_detections = []
labels = ["aceso", "apagado"]   # suas duas classes

NUM_QUEIMADORES = 4
IOU_LIMITE = 0.50 # Limite de IoU para considerar detecções de queimadores (apagados -> acesos) sobrepostas
queimadores_registrados = [] # Lista para armazenar os queimadores já registrados
queimadores_inicializados = False 

REFERENCIAS = {
    0: (480, 520),  # média aproximada das coordenadas do queimador sup. esquerdo
    1: (400, 260),  # média aproximada das coordenadas do queimador inf. esquerdo
    2: (300, 520),  # média aproximada das coordenadas do queimador sup. direito
    3: (260, 320),  # média aproximada das coordenadas do queimador inf. direito
}

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        #self.box = imx500.convert_inference_coords(coords, metadata, picam2)
        
        coords = np.array(coords, dtype=float).reshape(-1)

        if len(coords) != 4:
            print("WARNING: box inválido:", coords)

        self.box = coords.astype(int)



def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels
    
# Calcula o centro de uma box
def centro_box(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def atribuir_id_por_distancia(box):
    cx, cy = centro_box(box)

    melhor_id = None
    menor_dist = float("inf")

    for id_q, (rx, ry) in REFERENCIAS.items():
        dist = math.hypot(cx - rx, cy - ry)
        if dist < menor_dist:
            menor_dist = dist
            melhor_id = id_q

    return melhor_id


def convert_box_to_pixels(box, W, H):
    box = np.array(box)

    # Caso 2x2 → [[x1, y1], [x2, y2]]
    if box.ndim == 2 and box.shape == (2, 2):
        x1 = int(box[0, 0])
        y1 = int(box[0, 1])
        x2 = int(box[1, 0])
        y2 = int(box[1, 1])

    # Caso 1D → [x1, y1, x2, y2]
    elif box.ndim == 1 and box.size == 4:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

    else:
        print("⚠️ BOX INVÁLIDA:", box)
        return 0, 0, 0, 0

    w = x2 - x1
    h = y2 - y1

    return x1, y1, w, h



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()

def calcular_iou(boxA, boxB):
    
    # box = [x1, y1, x2, y2] 
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA) # Cálculo da área de interseção, se não houver interseção, será 0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter / float(areaA + areaB - inter + 1e-6) # Cálculo da Interseção sobre União (IoU) 1e-6 para evitar divisão por zero
    return iou

def registrar_quatro_apagados(deteccoes):
    apagados = [d for d in deteccoes if int(d.category) == 1]  # Filtra apenas os detectados como "apagado"

    if len(apagados) < NUM_QUEIMADORES: # Verifica se há menos de 4 queimadores apagados
        print(f"Aguardando detectar 4 apagados... {len(apagados)} detectados.")
        return False

    # Pega só os quatro primeiros
    usados = apagados[:NUM_QUEIMADORES]

    for d in usados:
        id_q = atribuir_id_por_distancia(d.box)

        queimadores_registrados.append({
            "id": id_q,
            "bbox": d.box.copy(),
            "estado": "apagado",
            "aceso": False
        })

    print("\n=== QUEIMADORES REGISTRADOS ===")
    for q in queimadores_registrados:
        print(q)

    print("================================\n")
    return True

def atualizar_estado_acesos(deteccoes):
    acesos = [d for d in deteccoes if int(d.category) == 0]  # Filtra apenas os detectados como "aceso"

    for queimador in queimadores_registrados:
        if queimador["aceso"]:
            continue

        for det in acesos:
            iou = calcular_iou(queimador["bbox"], det.box) # Calcula o IoU entre o queimador registrado (apagado) e a detecção atual (aceso)

            # Se o IoU for maior que o limite, marca como aceso
            if iou >= IOU_LIMITE:
                queimador["aceso"] = True
                queimador["estado"] = "aceso"
                print("Coordenadas do queimador aceso:", det.box)
                print(f"🔥 Queimador {queimador['id']} ACENDEU! (IoU={iou:.2f})")
                break

def todos_acesos():
    # Verifica se todos os queimadores estão acesos
    return all(q["aceso"] for q in queimadores_registrados)

if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (640, 640)},   # <<< A RESOLUÇÃO QUE VOCÊ QUER
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12
    )


    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    #picam2.pre_callback = draw_detections
    

    while True:
        last_results = parse_detections(picam2.capture_metadata())

        if not last_results:
            continue


        # ======================
        # 1) REGISTRAR 4 APAGADOS
        # ======================
        if not queimadores_inicializados:
            queimadores_inicializados = registrar_quatro_apagados(last_results)
            continue

        # ======================
        # 2) VERIFICAR TRANSIÇÃO APAGADO → ACESO
        # ======================
        atualizar_estado_acesos(last_results)

        # ======================
        # 3) VERIFICAR SE TODOS ACENDERAM
        # ======================
        if todos_acesos():
            print("\n🔥🔥🔥 TODOS OS QUEIMADORES ACENDERAM! 🔥🔥🔥\n")
            break
