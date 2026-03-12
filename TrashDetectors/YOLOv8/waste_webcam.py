import cv2
import torch
import time
from ultralytics import YOLO

# Para fallback manual via Hugging Face Hub
from huggingface_hub import hf_hub_download

# ----------------------------
# Configurações
# ----------------------------
HF_REPO_ID = "kendrickfff/waste-classification-yolov8-ken"
HF_WEIGHTS_FILE = "yolov8n-waste-12cls-best.pt"  # nome do .pt no repositório
CONF_THRES = 0.4
SOURCE = 0  # webcam padrão (0)

# ----------------------------
# Função para carregar o modelo
# ----------------------------
def load_model():
    """
    Tenta primeiro carregar usando o ID do Hugging Face diretamente.
    Se falhar (ultralytics antigo, etc.), baixa o .pt com hf_hub_download
    e carrega pelo caminho local.
    """
    print("[INFO] Tentando carregar modelo diretamente do Hugging Face via Ultralytics...")
    try:
        # ultralytics mais novo entende "user/model" e baixa do HF
        model = YOLO(HF_REPO_ID)
        print("[INFO] Modelo carregado diretamente do Hugging Face.")
        return model
    except Exception as e:
        print("[WARN] Falha ao carregar direto do HF:", e)
        print("[INFO] Baixando pesos .pt via huggingface_hub...")

        # Baixa só o arquivo de pesos .pt
        weights_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_WEIGHTS_FILE
        )
        print(f"[INFO] Pesos baixados em: {weights_path}")

        model = YOLO(weights_path)
        print("[INFO] Modelo carregado a partir do arquivo local.")
        return model


# ----------------------------
# Carregar modelo e configurar device
# ----------------------------
print("[INFO] Carregando modelo YOLOv8 de classificação de lixo...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Usando device: {device}")

model = load_model()
class_names = model.names  # dict: {id_classe: "nome"}

# ------------- NOVO: desabilitar classe "clothes" -------------
IGNORED_LABELS = {"clothes"}  # nomes a ignorar (pode adicionar mais)

# descobrir quais IDs correspondem a essas labels
ignored_class_ids = {
    cls_id
    for cls_id, name in class_names.items()
    if name.lower() in IGNORED_LABELS
}

# lista de classes permitidas (todas menos as ignoradas)
ALLOWED_CLASS_IDS = [cls_id for cls_id in class_names.keys()
                     if cls_id not in ignored_class_ids]

print("[INFO] Classes do modelo:")
for cid, name in class_names.items():
    print(f"  {cid}: {name}")
print(f"[INFO] Ignorando classe(s): {IGNORED_LABELS}")
print(f"[INFO] IDs permitidos para inferência: {ALLOWED_CLASS_IDS}")
# ---------------------------------------------------------------

# ----------------------------
# Abrir webcam
# ----------------------------
print("[INFO] Abrindo webcam...")
cap = cv2.VideoCapture(SOURCE, cv2.CAP_V4L2)

if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a webcam.")
    exit(1)

# Opcional: forçar resolução
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
frame_count = 0
fps = 0.0

print("[INFO] Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Falha ao ler frame da webcam.")
        break

    frame_count += 1

    # ----------------------------
    # Inferência
    # ----------------------------
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        device=device,
        verbose=False,
        classes=ALLOWED_CLASS_IDS
    )

    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())

            # Segurança extra: se por algum motivo vier "clothes", pula
            if cls_id in ignored_class_ids:
                continue  # <<< não desenha nada pra essa detecção

            # Coordenadas do bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Classe e confiança
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = class_names.get(cls_id, str(cls_id))

            text = f"{label} {conf:.2f}"

            # Desenhar retângulo e texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    # ----------------------------
    # Cálculo de FPS (média a cada 10 frames)
    # ----------------------------
    if frame_count >= 10:
        current_time = time.time()
        elapsed = current_time - prev_time
        fps = frame_count / elapsed
        prev_time = current_time
        frame_count = 0

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Mostrar janela
    cv2.imshow("Waste Classification - YOLOv8 (HF)", frame)

    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Encerrado.")
