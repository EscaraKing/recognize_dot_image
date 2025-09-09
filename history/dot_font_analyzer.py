import cv2
import numpy as np
import os
import math
import csv
import json
import tkinter as tk
from tkinter import Scale, Button, Label, Radiobutton, filedialog, Spinbox, Checkbutton, Text, END

# ------------------ 입력 폴더 ------------------
IMAGE_DIR = './manual_dot_image/images/'
image_files = [f for f in os.listdir(IMAGE_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.png'))]
if not image_files:
    print('No images found in images folder.')
    # exit()  # 학습 버튼에서 chars.csv 경로를 따로 고를 수 있으니 exit 생략

# ------------------ Dot Fonts (라벨 템플릿) ------------------
DOT_FONT_5x7 = {
    "A": ["01110","10001","10001","11111","10001","10001","10001"],
    "Y": ["10001","10001","01010","00100","00100","00100","00100"],
    "0": ["01110","10001","10011","10101","11001","10001","01110"],
    "1": ["00100","01100","00100","00100","00100","00100","01110"],
    "2": ["01110","10001","00001","00010","00100","01000","11111"],
    "3": ["01110","10001","00001","00110","00001","10001","01110"],
    "4": ["00010","00110","01010","10010","11111","00010","00010"],
    "5": ["11111","10000","11110","00001","00001","10001","01110"],
    "6": ["01110","10000","11110","10001","10001","10001","01110"],
    "7": ["11111","00001","00010","00100","01000","01000","01000"],
    "8": ["01110","10001","10001","01110","10001","10001","01110"],
    "9": ["01110","10001","10001","01111","00001","00001","01110"],
    " ": ["00000","00000","00000","00000","00000","00000","00000"]
}

DOT_FONT_7x9 = {
    "A": ["0011100","0100010","1000001","1000001","1111111","1000001","1000001","1000001","1000001"],
    "Y": ["1000001","1000001","0100010","0010100","0001000","0001000","0001000","0001000","0001000"],
    "0": ["0011100","0100010","1001001","1010101","1010101","1001001","0100010","0011100","0000000"],
    "1": ["0001000","0011000","0001000","0001000","0001000","0001000","0001000","0011100","0000000"],
    "2": ["0011100","0100010","0000010","0000100","0001000","0010000","0100000","1111110","0000000"],
    "3": ["0011100","0100010","0000010","0001100","0000010","0000010","0100010","0011100","0000000"],
    "4": ["0000100","0001100","0010100","0100100","1111110","0000100","0000100","0000100","0000000"],
    "5": ["1111110","1000000","1111100","0000010","0000010","0000010","1000010","0111100","0000000"],
    "6": ["0011100","0100000","1000000","1111100","1000010","1000010","1000010","0111100","0000000"],
    "7": ["1111110","0000010","0000100","0001000","0010000","0100000","0100000","0100000","0000000"],
    "8": ["0111100","1000010","1000010","0111100","1000010","1000010","1000010","0111100","0000000"],
    "9": ["0111100","1000010","1000010","0111110","0000010","0000010","0000010","0111100","0000000"],
    " ": ["0000000","0000000","0000000","0000000","0000000","0000000","0000000","0000000","0000000"]
}

# -----------------------------------------------------------
# Tkinter
# -----------------------------------------------------------
root = tk.Tk()
root.title("Dot ROI (Clustering + Grid + Training)")

log_text = Text(root, height=10)
def log(msg): log_text.insert(END, msg + "\n"); log_text.see(END)

# ------------------ 상태 변수 ------------------
img = None
img_height, img_width = 0, 0
if image_files:
    first_img_path = os.path.join(IMAGE_DIR, image_files[0])
    img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img_height, img_width = img.shape

# ROI
min_y_var = tk.IntVar(value=0 if img is None else 0)
max_y_var = tk.IntVar(value=(img_height - 1) if img is not None else 1)

# Threshold
thresh_var = tk.IntVar(value=127)
thresh_type_var = tk.StringVar(value="BINARY")  # BINARY, BINARY_INV, OTSU, OTSU_INV, ADAPTIVE_MEAN, ADAPTIVE_GAUSSIAN
adaptive_block_var = tk.IntVar(value=15)  # 홀수
adaptive_c_var = tk.IntVar(value=2)

# Blob params
min_area_var = tk.IntVar(value=50)
max_area_var = tk.IntVar(value=1200)
min_circularity_var = tk.DoubleVar(value=0.70)
max_circularity_var = tk.DoubleVar(value=1.00)

# Clustering (1D by X)
cluster_eps_var = tk.IntVar(value=20)      # px gap to merge
cluster_min_pts_var = tk.IntVar(value=3)   # minimum points per cluster

# Grid size (X × Y cells) 수동 모드에서 사용
grid_x_var = tk.IntVar(value=10)   # 열(cols)
grid_y_var = tk.IntVar(value=10)   # 행(rows)
CELL_VIS_PX = 10                   # 셀당 픽셀 크기 (저장시 모자이크)

# Auto grid toggle (박스 비율로 자동 결정)
auto_grid_var = tk.BooleanVar(value=False)

# Font choice for recognition
font_choice_var = tk.StringVar(value="5x7")  # "5x7" or "7x9"

# ------------------ Blob Detector ------------------
def build_detector(min_area, max_area, min_circ, max_circ):
    min_area = max(1.0, float(min_area))
    max_area = max(min_area, float(max_area))
    min_circ = max(0.0, min(1.0, float(min_circ)))
    max_circ = max(min_circ, min(1.0, float(max_circ)))

    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 0; p.maxThreshold = 255
    p.filterByArea = True; p.minArea = min_area; p.maxArea = max_area
    p.filterByCircularity = True; p.minCircularity = min_circ; p.maxCircularity = max_circ
    p.filterByInertia = False; p.filterByConvexity = False; p.filterByColor = False
    if cv2.__version__.split('.')[0] == '2':
        return cv2.SimpleBlobDetector(p)
    return cv2.SimpleBlobDetector_create(p)

# ------------------ Threshold ------------------
def get_thresholded(gray):
    ttype = thresh_type_var.get()
    t = thresh_var.get()
    if ttype == "BINARY":
        return cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)[1]
    if ttype == "BINARY_INV":
        return cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)[1]
    if ttype == "OTSU":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if ttype == "OTSU_INV":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    block = adaptive_block_var.get()
    if block % 2 == 0: block += 1
    C = adaptive_c_var.get()
    if ttype == "ADAPTIVE_MEAN":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, block, C)
    if ttype == "ADAPTIVE_GAUSSIAN":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, C)
    return cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)[1]

# ------------------ 1D 클러스터링 (X축 기준) ------------------
def cluster_1d_by_x(xs, eps, min_pts):
    xs = np.asarray(xs)
    if len(xs) == 0:
        return []
    order = np.argsort(xs)
    xs_sorted = xs[order]
    clusters = []
    start = 0
    for i in range(1, len(xs_sorted)):
        if xs_sorted[i] - xs_sorted[i-1] > eps:
            group_idx_sorted = order[start:i]
            if len(group_idx_sorted) >= min_pts:
                clusters.append(list(group_idx_sorted))
            start = i
    group_idx_sorted = order[start:len(xs_sorted)]
    if len(group_idx_sorted) >= min_pts:
        clusters.append(list(group_idx_sorted))
    return clusters

# ------------------ 색상 팔레트 ------------------
def color_palette():
    return [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (0, 128, 255), (255, 0, 128), (128, 255, 0), (128, 0, 255),
    ]

# ------------------ Auto Grid: 박스 비율 기반 (<10/axis 기본, 원하시면 6 등으로 바꿨던 곳) ------------------
def auto_grid_from_bbox(w, h):
    w = max(1, int(w)); h = max(1, int(h))
    g = math.gcd(w, h)
    gx = max(1, w // g)
    gy = max(1, h // g)
    max_allowed = 9   # ← 여기 값을 6으로 바꾸면 각 축 최대 6
    if gx > max_allowed or gy > max_allowed:
        scale = math.ceil(max(gx, gy) / max_allowed)
        gx = max(2, gx // scale)
        gy = max(2, gy // scale)
    gx = max(2, gx); gy = max(2, gy)
    return gx, gy

# ------------------ 클러스터 → Grid (cols=X, rows=Y) ------------------
def cluster_to_binary_grid(cluster_pts, grid_x, grid_y):
    grid_x = max(2, int(grid_x))
    grid_y = max(2, int(grid_y))
    if len(cluster_pts) == 0:
        return 255 * np.ones((grid_y, grid_x), dtype=np.uint8)

    min_x, min_y = np.min(cluster_pts[:,0]), np.min(cluster_pts[:,1])
    max_x, max_y = np.max(cluster_pts[:,0]), np.max(cluster_pts[:,1])
    w = max(1, max_x - min_x + 1)
    h = max(1, max_y - min_y + 1)

    x_edges = min_x + np.linspace(0, w, grid_x + 1)
    y_edges = min_y + np.linspace(0, h, grid_y + 1)

    grid = 255 * np.ones((grid_y, grid_x), dtype=np.uint8)
    xs = cluster_pts[:,0]; ys = cluster_pts[:,1]
    gx = np.clip(np.searchsorted(x_edges, xs, side='right') - 1, 0, grid_x - 1)
    gy = np.clip(np.searchsorted(y_edges, ys, side='right') - 1, 0, grid_y - 1)
    grid[gy, gx] = 0
    return grid

# ------------------ 검출 + 클러스터 + 시각화 ------------------
def detect_and_draw_with_clusters(base_gray, min_y, max_y, draw_bbox=True):
    threshed = get_thresholded(base_gray)
    roi = threshed[min_y:max_y, :]

    detector = build_detector(min_area_var.get(), max_area_var.get(),
                              min_circularity_var.get(), max_circularity_var.get())
    kps = detector.detect(roi)

    pts = []
    radii = []
    for kp in kps:
        x = int(kp.pt[0]); y = int(kp.pt[1]) + min_y
        r = max(1, int(kp.size / 2))
        pts.append((x, y)); radii.append(r)
    pts = np.array(pts, dtype=np.int32) if len(pts) else np.zeros((0,2), dtype=np.int32)
    radii = np.array(radii, dtype=np.int32) if len(radii) else np.zeros((0,), dtype=np.int32)

    vis_bgr = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
    gray_color = (180, 180, 180)
    for i in range(len(pts)):
        cv2.circle(vis_bgr, (int(pts[i,0]), int(pts[i,1])), int(radii[i]), gray_color, 2)

    eps = cluster_eps_var.get(); min_pts = cluster_min_pts_var.get()
    xs = pts[:, 0] if len(pts) else np.array([], dtype=np.int32)
    clusters = cluster_1d_by_x(xs, eps=eps, min_pts=min_pts)

    palette = color_palette()
    for ci, idx_list in enumerate(clusters):
        color = palette[ci % len(palette)]
        cluster_pts = pts[idx_list]
        for gi in idx_list:
            cx, cy = int(pts[gi,0]), int(pts[gi,1])
            rr = 2
            cv2.circle(vis_bgr, (cx, cy), rr, color, 2)
        if draw_bbox and len(cluster_pts):
            min_x, min_yc = np.min(cluster_pts[:,0]), np.min(cluster_pts[:,1])
            max_x, max_yc = np.max(cluster_pts[:,0]), np.max(cluster_pts[:,1])
            cv2.rectangle(vis_bgr, (int(min_x), int(min_yc)),
                          (int(max_x), int(max_yc)), color, 2)

    # ROI 가이드 / HUD
    cv2.line(vis_bgr, (0, min_y), (img_width, min_y), (0,255,0), 2)
    cv2.line(vis_bgr, (0, max_y), (img_width, max_y), (0,0,255), 2)
    hud = f"Clusters: {len(clusters)} | AutoGrid: {'ON' if auto_grid_var.get() else 'OFF'}"
    cv2.putText(vis_bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return vis_bgr, clusters, pts

# ------------------ 미리보기 ------------------
def update_image(*_):
    # 수동 입력창 활성/비활성 토글
    state = 'disabled' if auto_grid_var.get() else 'normal'
    grid_x_spin.config(state=state)
    grid_y_spin.config(state=state)

    if img is None:
        return
    min_y = min_y_var.get(); max_y = max_y_var.get()
    if min_y >= max_y: max_y = min_y + 1; max_y_var.set(max_y)
    vis, clusters, pts = detect_and_draw_with_clusters(img, min_y, max_y, draw_bbox=True)
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preview', 1200, 900)
    cv2.imshow('Preview', vis); cv2.waitKey(1)

# ------------------ 분석/저장 (기존) ------------------
def analyze_all():
    save_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")
    if not save_dir:
        log("저장 폴더가 선택되지 않았습니다."); return

    if not image_files:
        log("IMAGE_DIR 에 이미지가 없습니다."); return

    min_y = min_y_var.get()
    max_y = max_y_var.get()
    manual_grid_x = max(2, int(grid_x_var.get()))
    manual_grid_y = max(2, int(grid_y_var.get()))
    cell_px = CELL_VIS_PX
    use_auto = auto_grid_var.get()

    for fname in image_files:
        path = os.path.join(IMAGE_DIR, fname)
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            log(f'Failed to load {fname}')
            continue

        vis, clusters, pts = detect_and_draw_with_clusters(g, min_y, max_y, draw_bbox=True)
        clustered_path = os.path.join(save_dir, f'clustered_{fname}')
        cv2.imwrite(clustered_path, vis)
        log(f"Saved clustered image: {clustered_path}")

        # 클러스터 → 그리드 저장
        for ci, idx_list in enumerate(clusters):
            cluster_pts = pts[idx_list]
            if len(cluster_pts) == 0:
                continue
            # 박스 계산
            min_x, min_yc = np.min(cluster_pts[:,0]), np.min(cluster_pts[:,1])
            max_x, max_yc = np.max(cluster_pts[:,0]), np.max(cluster_pts[:,1])
            w = max_x - min_x + 1
            h = max_yc - min_yc + 1

            if use_auto:
                grid_x, grid_y = auto_grid_from_bbox(w, h)
            else:
                grid_x, grid_y = manual_grid_x, manual_grid_y

            grid = cluster_to_binary_grid(cluster_pts, grid_x, grid_y)
            mosaic = cv2.resize(grid, (grid_x*cell_px, grid_y*cell_px), interpolation=cv2.INTER_NEAREST)
            grid_path = os.path.join(save_dir, f'grid_{grid_y}x{grid_x}_cluster{ci}_{fname}')
            cv2.imwrite(grid_path, mosaic)

    cv2.destroyAllWindows()

# ===========================================================
# ==================== 학습 / 평가 파트 ======================
# ===========================================================

def font_dict_from_choice():
    return DOT_FONT_5x7 if font_choice_var.get() == "5x7" else DOT_FONT_7x9

def font_pattern_to_binary(char, font_dict):
    """
    폰트 패턴을 0/1 numpy로 변환 (1=점 있음, 0=없음)
    shape: (rows, cols)
    """
    grid_lines = font_dict.get(char, font_dict[" "])
    rows = len(grid_lines)
    cols = len(grid_lines[0])
    arr = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            arr[r, c] = 1 if grid_lines[r][c] == "1" else 0
    return arr

def resize_binary_pattern(arr, target_h, target_w):
    """
    최근접 보간으로 이진 패턴 리샘플 → (target_h, target_w) 0/1
    """
    src_h, src_w = arr.shape
    if src_h == target_h and src_w == target_w:
        return arr.copy()
    # OpenCV는 0/255 기준이 편하므로 변환
    img = (arr * 255).astype(np.uint8)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.uint8)

def predict_char_from_grid(grid_255, font_dict):
    """
    grid_255: (H,W) uint8, 0=점있음, 255=없음
    → 각 문자 템플릿을 동일크기로 리샘플하여 해밍거리 최소 문자 반환
    return: (pred_char, confidence)  confidence=1 - (min_dist/total_cells)
    """
    H, W = grid_255.shape
    # grid를 0/1로 변환 (1=점 있음)
    grid_bin = (grid_255 == 0).astype(np.uint8)

    best_char = "?"
    best_dist = 10**9
    total = H * W

    font_dict_local = font_dict
    for ch in font_dict_local.keys():
        if ch == " ":
            continue
        tmpl = font_pattern_to_binary(ch, font_dict_local)  # 0/1
        tmpl_r = resize_binary_pattern(tmpl, H, W)          # 0/1
        # 해밍 거리
        dist = np.count_nonzero(grid_bin ^ tmpl_r)
        if dist < best_dist:
            best_dist = dist
            best_char = ch

    conf = 1.0 - (best_dist / max(1, total))
    return best_char, conf

def load_chars_csv(chars_csv_path):
    """
    chars.csv 로드 → {image_filename: [ {char, box, char_index}, ... ] }
    - box: (x1,y1,x2,y2)
    - image_filename: 파일명만 매칭하도록 정규화
    """
    mapping = {}
    with open(chars_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row["image_path"]
            img_fn = os.path.basename(image_path)
            ch = row["char_label"]
            x1 = int(row["char_box_x1"]); y1 = int(row["char_box_y1"])
            x2 = int(row["char_box_x2"]); y2 = int(row["char_box_y2"])
            char_index = int(row.get("char_index", 0))
            item = {"char": ch, "box": (x1,y1,x2,y2), "char_index": char_index}
            mapping.setdefault(img_fn, []).append(item)
    # 정답 문자들을 x 중심 기준으로 정렬(왼→오) → 매칭 안정화
    for k in mapping.keys():
        mapping[k].sort(key=lambda it: ( (it["box"][0]+it["box"][2]) / 2.0 ))
    return mapping

def center_and_width(box):
    x1,y1,x2,y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, x2 - x1)
    return (cx, cy, w)

def evaluate_once_on_dataset(chars_map, config):
    """
    주어진 설정(config)으로 한번 전체 평가
    return: (acc, total, correct, per_image_results)
    """
    # 설정 적용
    thresh_type_var.set(config["thresh_type"])
    thresh_var.set(config.get("thresh", thresh_var.get()))
    # adaptive는 필요 시 추가 가능
    # blob 파라미터/클러스터 파라미터도 config로 확장 가능

    font_dict = font_dict_from_choice()

    total = 0
    correct = 0
    per_image = {}

    # 평가 대상: chars_map 에 등장하는 이미지들만
    eval_files = list(chars_map.keys())

    for img_fn in eval_files:
        # 이미지 읽기(우선 IMAGE_DIR에서 찾고, 없으면 chars.csv 경로 그대로 시도)
        path1 = os.path.join(IMAGE_DIR, img_fn)
        if os.path.exists(path1):
            g = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            img_path_used = path1
        else:
            # chars.csv의 절대/상대 경로가 다를 수 있어 파일명만 일치시킴
            log(f"[경고] {img_fn} 를 {IMAGE_DIR}에서 찾을 수 없음. 스킵.")
            continue
        if g is None:
            log(f"Failed to load {img_fn}"); continue

        min_y = min_y_var.get(); max_y = max_y_var.get()
        vis, clusters, pts = detect_and_draw_with_clusters(g, min_y, max_y, draw_bbox=False)

        # 예측용 클러스터 리스트(박스, 그리드, 예측문자)
        predicted = []
        used_indices = set()
        manual_grid_x = max(2, int(grid_x_var.get()))
        manual_grid_y = max(2, int(grid_y_var.get()))
        use_auto = auto_grid_var.get()

        for ci, idx_list in enumerate(clusters):
            cluster_pts = pts[idx_list]
            if len(cluster_pts) == 0:
                continue
            # 박스 계산
            min_x, min_yc = np.min(cluster_pts[:,0]), np.min(cluster_pts[:,1])
            max_x, max_yc = np.max(cluster_pts[:,0]), np.max(cluster_pts[:,1])
            w = max_x - min_x + 1
            h = max_yc - min_yc + 1

            if use_auto:
                grid_x, grid_y = auto_grid_from_bbox(w, h)
            else:
                grid_x, grid_y = manual_grid_x, manual_grid_y

            grid = cluster_to_binary_grid(cluster_pts, grid_x, grid_y)  # 0/255
            pred_char, conf = predict_char_from_grid(grid, font_dict)
            predicted.append({
                "box": (int(min_x), int(min_yc), int(max_x), int(max_yc)),
                "pred": pred_char,
                "conf": conf,
                "center": ((min_x+max_x)/2.0, (min_yc+max_yc)/2.0),
                "width": max(1.0, w)
            })

        # 정답
        gts = chars_map.get(img_fn, [])
        # 안정적 매칭: 예측과 정답을 양쪽 x중심 오름차순으로 정렬한 뒤 같은 인덱스끼리 매칭
        predicted.sort(key=lambda it: it["center"][0])
        # gts는 이미 load 시 정렬됨

        n = min(len(predicted), len(gts))
        img_total = n
        img_correct = 0
        for i in range(n):
            p = predicted[i]
            gt = gts[i]
            # 라벨 비교
            label_ok = (p["pred"] == gt["char"])

            # 위치 비교: 중심 거리 > 정답 박스 폭 이면 실패
            pcx, pcy = p["center"]
            gcx, gcy, g_w = center_and_width(gt["box"])
            dist = math.sqrt((pcx-gcx)**2 + (pcy-gcy)**2)
            pos_ok = (dist <= g_w)

            ok = (label_ok and pos_ok)
            total += 1
            if ok: correct += 1
        per_image[img_fn] = {"total": img_total, "correct": img_correct}

    acc = (correct / total) if total > 0 else 0.0
    return acc, total, correct, per_image

def train_until_accuracy():
    # chars.csv 선택
    chars_csv_path = filedialog.askopenfilename(
        title="chars.csv 선택", filetypes=[("CSV","*.csv")]
    )
    if not chars_csv_path:
        log("chars.csv 가 선택되지 않았습니다."); return

    # chars.csv 로드
    try:
        chars_map = load_chars_csv(chars_csv_path)
    except Exception as e:
        log(f"chars.csv 로드 실패: {e}")
        return

    # 폰트 선택(템플릿)
    font_used = font_choice_var.get()
    log(f"[학습] 템플릿 폰트: {font_used}")

    # 간단한 그리드서치 후보
    thresh_candidates = [
        ("BINARY", 80), ("BINARY", 100), ("BINARY", 127),
        ("BINARY_INV", 127),
        ("OTSU", None), ("OTSU_INV", None),
        ("ADAPTIVE_MEAN", None), ("ADAPTIVE_GAUSSIAN", None),
    ]

    best_acc = -1.0
    best_conf = None

    for ttype, tval in thresh_candidates:
        conf = {"thresh_type": ttype}
        if tval is not None:
            conf["thresh"] = tval

        acc, total, correct, _ = evaluate_once_on_dataset(chars_map, conf)
        log(f"[평가] {ttype} {tval if tval is not None else ''} → acc={acc*100:.2f}% ({correct}/{total})")

        if acc > best_acc:
            best_acc = acc
            best_conf = conf

        if acc >= 0.90:
            log("[종료] 정확도 90% 이상 달성!")
            break

    if best_conf is None:
        log("평가 실패(유효한 설정 없음)."); return

    # 최종 설정을 UI에 반영
    thresh_type_var.set(best_conf["thresh_type"])
    if "thresh" in best_conf:
        thresh_var.set(best_conf["thresh"])

    log(f"[최종] Best acc={best_acc*100:.2f}% with {best_conf}")

# ------------------ UI ------------------
Label(root, text="ROI - min_y").pack()
Scale(root, from_=0, to=(img_height-1 if img is not None else 1), orient='horizontal', variable=min_y_var, command=update_image).pack(fill='x')

Label(root, text="ROI - max_y").pack()
Scale(root, from_=0, to=(img_height-1 if img is not None else 1), orient='horizontal', variable=max_y_var, command=update_image).pack(fill='x')

Label(root, text="Threshold Type").pack()
frame = tk.Frame(root); frame.pack()
for key, text in [("BINARY","Binary"), ("BINARY_INV","Binary Inv"),
                  ("OTSU","Otsu"), ("OTSU_INV","Otsu Inv"),
                  ("ADAPTIVE_MEAN","Adaptive Mean"),
                  ("ADAPTIVE_GAUSSIAN","Adaptive Gaussian")]:
    Radiobutton(frame, text=text, value=key, variable=thresh_type_var, command=update_image).pack(side='left')

Label(root, text="threshold (Binary계열)").pack()
Scale(root, from_=0, to=255, orient='horizontal', variable=thresh_var, command=update_image).pack(fill='x')

Label(root, text="Adaptive block size (odd)").pack()
Scale(root, from_=3, to=51, orient='horizontal', variable=adaptive_block_var, command=update_image).pack(fill='x')

Label(root, text="Adaptive C").pack()
Scale(root, from_=-20, to=20, orient='horizontal', variable=adaptive_c_var, command=update_image).pack(fill='x')

Label(root, text="min_area").pack()
Scale(root, from_=1, to=5000, orient='horizontal', variable=min_area_var, command=update_image).pack(fill='x')

Label(root, text="max_area").pack()
Scale(root, from_=10, to=10000, orient='horizontal', variable=max_area_var, command=update_image).pack(fill='x')

Label(root, text="Cluster gap eps (px)").pack()
Scale(root, from_=1, to=200, orient='horizontal', variable=cluster_eps_var, command=update_image).pack(fill='x')

Label(root, text="Cluster min points").pack()
Scale(root, from_=1, to=20, orient='horizontal', variable=cluster_min_pts_var, command=update_image).pack(fill='x')

# ---- Auto Grid 토글 ----
Checkbutton(root, text="Auto grid from bbox ratio (<10 per axis)", variable=auto_grid_var,
            command=update_image).pack(pady=(8, 0))

# ---- Grid 크기: X, Y Spinbox (수동 모드 전용) ----
Label(root, text="Grid size (X cells / cols)").pack()
grid_x_spin = Spinbox(root, from_=2, to=64, textvariable=grid_x_var, width=5, command=update_image)
grid_x_spin.pack()
Label(root, text="Grid size (Y cells / rows)").pack()
grid_y_spin = Spinbox(root, from_=2, to=64, textvariable=grid_y_var, width=5, command=update_image)
grid_y_spin.pack()

# ---- 인식 템플릿 폰트 선택 ----
Label(root, text="Recognition Font (template)").pack()
frame2 = tk.Frame(root); frame2.pack()
Radiobutton(frame2, text="5x7", value="5x7", variable=font_choice_var).pack(side='left')
Radiobutton(frame2, text="7x9", value="7x9", variable=font_choice_var).pack(side='left')

Button(root, text="분석 시작 (폴더 저장)", command=analyze_all).pack(pady=6)
Button(root, text="학습 시작 (chars.csv와 비교, 90%까지)", command=train_until_accuracy).pack(pady=6)

Label(root, text="Logs").pack()
log_text.pack(fill='both', expand=True)

# 초기 미리보기
update_image()
root.mainloop()
