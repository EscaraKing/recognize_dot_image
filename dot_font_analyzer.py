# dot_font_analyzer.py
import cv2
import numpy as np
import os
import math
import csv
import tkinter as tk
from tkinter import Scale, Button, Label, Radiobutton, filedialog, Spinbox, Checkbutton, Text, END

# ===================== 입력 이미지 폴더 =====================
IMAGE_DIR = './manual_dot_image/images/'
image_files = [f for f in os.listdir(IMAGE_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print('No images found in manual_dot_image/images (IMAGE_DIR).')

# 첫 이미지 로드(미리보기 초기화용)
img = None
img_height, img_width = 1, 1
if image_files:
    first_img_path = os.path.join(IMAGE_DIR, image_files[0])
    img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img_height, img_width = img.shape

# ===================== Tkinter UI =====================
root = tk.Tk()
root.title("Dot Font Analyzer (ROI + Clustering + Grid + clusters.csv)")

log_text = Text(root, height=10)
def log(msg): log_text.insert(END, msg + "\n"); log_text.see(END)

# ----- 상태 변수 -----
# ROI
min_y_var = tk.IntVar(value=0)
max_y_var = tk.IntVar(value=img_height - 1 if img is not None else 1)

# Threshold
thresh_var = tk.IntVar(value=127)
thresh_type_var = tk.StringVar(value="BINARY")  # BINARY, BINARY_INV, OTSU, OTSU_INV, ADAPTIVE_MEAN, ADAPTIVE_GAUSSIAN
adaptive_block_var = tk.IntVar(value=15)  # 홀수
adaptive_c_var = tk.IntVar(value=2)

# Blob (SimpleBlobDetector)
min_area_var = tk.IntVar(value=50)
max_area_var = tk.IntVar(value=1200)
min_circularity_var = tk.DoubleVar(value=0.70)
max_circularity_var = tk.DoubleVar(value=1.00)

# Clustering (1D by X center)
cluster_eps_var = tk.IntVar(value=20)      # 같은 클러스터로 묶을 최대 간격(px)
cluster_min_pts_var = tk.IntVar(value=3)   # 최소 점 개수

# Grid
grid_x_var = tk.IntVar(value=10)   # 수동 cols
grid_y_var = tk.IntVar(value=10)   # 수동 rows
CELL_VIS_PX = 10                   # 모자이크 저장 시 셀당 픽셀 크기

auto_grid_var = tk.BooleanVar(value=True)  # 자동 그리드 기본 ON (축 최대 6)

# ===================== 유틸/핵심 로직 =====================
def build_detector(min_area, max_area, min_circ, max_circ):
    """OpenCV SimpleBlobDetector 구성 (면적/원형도 기준)."""
    min_area = max(1.0, float(min_area))
    max_area = max(min_area, float(max_area))
    min_circ = max(0.0, min(1.0, float(min_circ)))
    max_circ = max(min_circ, min(1.0, float(max_circ)))

    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 0; p.maxThreshold = 255
    p.filterByArea = True; p.minArea = min_area; p.maxArea = max_area
    p.filterByCircularity = True; p.minCircularity = min_circ; p.maxCircularity = max_circ
    p.filterByInertia = False; p.filterByConvexity = False; p.filterByColor = False

    # OpenCV 버전 호환
    if cv2.__version__.split('.')[0] == '2':
        return cv2.SimpleBlobDetector(p)
    return cv2.SimpleBlobDetector_create(p)

def get_thresholded(gray):
    """선택한 방식으로 이진화."""
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

def cluster_1d_by_x(xs, eps, min_pts):
    """X좌표(중심) 기준 간격 eps 이하로 연결되는 점들을 클러스터로 묶음."""
    xs = np.asarray(xs)
    if len(xs) == 0: return []
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

def color_palette():
    return [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (0, 128, 255), (255, 0, 128), (128, 255, 0), (128, 0, 255),
    ]

def auto_grid_from_bbox(w, h):
    """
    박스 비율을 최대공약수로 축소 → 각 축 최대 6이 되도록 스케일.
    최소 2셀 보장.
    """
    w = max(1, int(w)); h = max(1, int(h))
    g = math.gcd(w, h)
    gx = max(1, w // g)
    gy = max(1, h // g)
    max_allowed = 6  # ⬅ 요청 반영: 축 최대 6
    if gx > max_allowed or gy > max_allowed:
        scale = math.ceil(max(gx, gy) / max_allowed)
        gx = max(2, gx // scale)
        gy = max(2, gy // scale)
    gx = max(2, gx)
    gy = max(2, gy)
    return gx, gy

def cluster_to_binary_grid(cluster_pts, grid_x, grid_y):
    """
    클러스터 점 좌표를 grid_x×grid_y 셀로 매핑(점 존재 셀=0, 없으면 255).
    """
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

def detect_and_draw_with_clusters(base_gray, min_y, max_y, draw_bbox=True):
    """
    이진화 → blob 검출 → X축 1D 클러스터링 → 시각화 이미지와 (클러스터,점) 반환
    """
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

    # 1) 모든 점 회색으로
    gray_color = (180, 180, 180)
    for i in range(len(pts)):
        cv2.circle(vis_bgr, (int(pts[i,0]), int(pts[i,1])), int(max(1, radii[i]//2)), gray_color, 2)

    # 2) X축 클러스터링
    eps = cluster_eps_var.get(); min_pts = cluster_min_pts_var.get()
    xs = pts[:, 0] if len(pts) else np.array([], dtype=np.int32)
    clusters = cluster_1d_by_x(xs, eps=eps, min_pts=min_pts)

    # 3) 클러스터 색상/박스
    palette = color_palette()
    for ci, idx_list in enumerate(clusters):
        color = palette[ci % len(palette)]
        if len(idx_list) == 0: continue
        cluster_pts = pts[idx_list]
        # 포인트 강조
        for gi in idx_list:
            cx, cy = int(pts[gi,0]), int(pts[gi,1])
            cv2.circle(vis_bgr, (cx, cy), 2, color, 2)
        # 박스
        if draw_bbox:
            min_x, min_yc = np.min(cluster_pts[:,0]), np.min(cluster_pts[:,1])
            max_x, max_yc = np.max(cluster_pts[:,0]), np.max(cluster_pts[:,1])
            cv2.rectangle(vis_bgr, (int(min_x), int(min_yc)),
                          (int(max_x), int(max_yc)), color, 2)

    # 4) ROI 라인 / HUD
    cv2.line(vis_bgr, (0, min_y), (img_width, min_y), (0,255,0), 2)
    cv2.line(vis_bgr, (0, max_y), (img_width, max_y), (0,0,255), 2)
    hud = f"Clusters: {len(clusters)} | AutoGrid: {'ON' if auto_grid_var.get() else 'OFF'}"
    cv2.putText(vis_bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return vis_bgr, clusters, pts

# ===================== 미리보기 =====================
def update_image(*_):
    # 수동 그리드 입력 활성/비활성
    state = 'disabled' if auto_grid_var.get() else 'normal'
    grid_x_spin.config(state=state)
    grid_y_spin.config(state=state)

    if img is None:
        return
    min_y = min_y_var.get(); max_y = max_y_var.get()
    if min_y >= max_y:
        max_y = min_y + 1
        max_y_var.set(max_y)

    vis, clusters, pts = detect_and_draw_with_clusters(img, min_y, max_y, draw_bbox=True)
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preview', 1200, 900)
    cv2.imshow('Preview', vis); cv2.waitKey(1)

# ===================== 분석/저장 =====================
def analyze_all():
    """
    - 저장 폴더 선택
    - clustered_*.png 저장
    - grid_*_cluster*.png 저장
    - clusters.csv (메타: grid 경로, 그리드 크기, 박스, 중심) 저장/append
    """
    save_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")
    if not save_dir:
        log("저장 폴더가 선택되지 않았습니다."); return

    # clusters.csv 준비
    clusters_csv = os.path.join(save_dir, "clusters.csv")
    write_header = not os.path.exists(clusters_csv)
    f_meta = open(clusters_csv, "a", newline="", encoding="utf-8")
    meta_writer = csv.writer(f_meta)
    if write_header:
        meta_writer.writerow([
            "image_filename","cluster_index",
            "grid_path","grid_rows","grid_cols",
            "box_x1","box_y1","box_x2","box_y2",
            "center_x","center_y"
        ])

    min_y = min_y_var.get(); max_y = max_y_var.get()
    manual_grid_x = max(2, int(grid_x_var.get()))
    manual_grid_y = max(2, int(grid_y_var.get()))
    use_auto = auto_grid_var.get()
    cell_px = CELL_VIS_PX

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

        for ci, idx_list in enumerate(clusters):
            cluster_pts = pts[idx_list]
            if len(cluster_pts) == 0:
                continue

            min_x, min_yc = np.min(cluster_pts[:,0]), np.min(cluster_pts[:,1])
            max_x, max_yc = np.max(cluster_pts[:,0]), np.max(cluster_pts[:,1])
            w = max_x - min_x + 1
            h = max_yc - min_yc + 1

            # 그리드 크기 결정
            if use_auto:
                grid_x, grid_y = auto_grid_from_bbox(w, h)   # cols=X, rows=Y
            else:
                grid_x, grid_y = manual_grid_x, manual_grid_y

            # 바이너리 그리드 및 모자이크 저장
            grid = cluster_to_binary_grid(cluster_pts, grid_x, grid_y)  # (rows, cols) 0/255
            mosaic = cv2.resize(grid, (grid_x*cell_px, grid_y*cell_px), interpolation=cv2.INTER_NEAREST)
            grid_path = os.path.join(save_dir, f'grid_{grid_y}x{grid_x}_cluster{ci}_{fname}')
            cv2.imwrite(grid_path, mosaic)

            # 메타(=학습/평가용) 기록
            cx = (min_x + max_x) / 2.0
            cy = (min_yc + max_yc) / 2.0
            meta_writer.writerow([
                fname, ci,
                grid_path, grid_y, grid_x,
                int(min_x), int(min_yc), int(max_x), int(max_yc),
                cx, cy
            ])

    f_meta.close()
    cv2.destroyAllWindows()
    log(f"clusters.csv updated: {clusters_csv}")

# ===================== UI 구성 =====================
Label(root, text="ROI - min_y").pack()
Scale(root, from_=0, to=(img_height-1 if img is not None else 1), orient='horizontal',
      variable=min_y_var, command=update_image).pack(fill='x')

Label(root, text="ROI - max_y").pack()
Scale(root, from_=0, to=(img_height-1 if img is not None else 1), orient='horizontal',
      variable=max_y_var, command=update_image).pack(fill='x')

Label(root, text="Threshold Type").pack()
frame_t = tk.Frame(root); frame_t.pack()
for key, text in [("BINARY","Binary"), ("BINARY_INV","Binary Inv"),
                  ("OTSU","Otsu"), ("OTSU_INV","Otsu Inv"),
                  ("ADAPTIVE_MEAN","Adaptive Mean"), ("ADAPTIVE_GAUSSIAN","Adaptive Gaussian")]:
    Radiobutton(frame_t, text=text, value=key, variable=thresh_type_var, command=update_image).pack(side='left')

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

Label(root, text="min_circularity").pack()
Scale(root, from_=0, to=100, orient='horizontal',
      variable=tk.IntVar(value=int(min_circularity_var.get()*100)),
      command=lambda v: (min_circularity_var.set(float(v)/100), update_image())).pack(fill='x')

Label(root, text="max_circularity").pack()
Scale(root, from_=0, to=100, orient='horizontal',
      variable=tk.IntVar(value=int(max_circularity_var.get()*100)),
      command=lambda v: (max_circularity_var.set(float(v)/100), update_image())).pack(fill='x')

Label(root, text="Cluster gap eps (px)").pack()
Scale(root, from_=1, to=200, orient='horizontal', variable=cluster_eps_var, command=update_image).pack(fill='x')

Label(root, text="Cluster min points").pack()
Scale(root, from_=1, to=20, orient='horizontal', variable=cluster_min_pts_var, command=update_image).pack(fill='x')

Checkbutton(root, text="Auto grid from bbox ratio (axis ≤ 6)", variable=auto_grid_var,
            command=update_image).pack(pady=(8, 0))

Label(root, text="Grid size (X cells / cols)").pack()
grid_x_spin = Spinbox(root, from_=2, to=64, textvariable=grid_x_var, width=5, command=update_image)
grid_x_spin.pack()
Label(root, text="Grid size (Y cells / rows)").pack()
grid_y_spin = Spinbox(root, from_=2, to=64, textvariable=grid_y_var, width=5, command=update_image)
grid_y_spin.pack()

Button(root, text="분석 시작 (그리드/메타 저장)", command=analyze_all).pack(pady=8)

Label(root, text="Logs").pack()
log_text.pack(fill='both', expand=True)

# 초기 미리보기
update_image()
root.mainloop()
