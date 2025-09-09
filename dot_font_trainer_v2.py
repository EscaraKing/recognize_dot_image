# dot_font_trainer.py
# Tkinter GUI + 전체 스크롤 + 로그 스크롤 / 그리드 기반 학습 (analysis_dir/clusters.csv 사용)

import os, math, random, threading, queue
import pandas as pd
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =========================
# 공통: 라벨 매핑 (0-9 + A-Z)
# =========================
VOCAB = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z')+1)]
LABEL2ID = {ch:i for i,ch in enumerate(VOCAB)}
ID2LABEL = {i:ch for ch,i in LABEL2ID.items()}

def label_to_id(ch):
    ch = str(ch).upper()
    if ch not in LABEL2ID: raise ValueError(f"Unsupported label: {ch}")
    return LABEL2ID[ch]

# =========================
# 데이터 로더 (chars/clusters)
# =========================
def load_chars_csv(chars_csv):
    df = pd.read_csv(chars_csv)
    df["image_filename"] = df["image_path"].apply(lambda p: os.path.basename(str(p)))
    df["gcx"] = (df["char_box_x1"] + df["char_box_x2"]) / 2.0
    df["gcy"] = (df["char_box_y1"] + df["char_box_y2"]) / 2.0
    df["gw"]  = (df["char_box_x2"] - df["char_box_x1"]).clip(lower=1.0)
    df = df[df["char_label"].astype(str).str.upper().isin(VOCAB)].reset_index(drop=True)
    return df

def load_clusters_csv(clusters_csv):
    cdf = pd.read_csv(clusters_csv)
    required = {
        "image_filename","cluster_index","grid_path","grid_rows","grid_cols",
        "box_x1","box_y1","box_x2","box_y2","center_x","center_y"
    }
    missing = required - set(cdf.columns)
    if missing:
        raise ValueError(f"Missing columns in clusters.csv: {missing}")
    return cdf

def pair_by_image_and_order(chars_df, clusters_df):
    rows = []
    for img_fn, gt_grp in chars_df.groupby("image_filename"):
        if img_fn not in clusters_df["image_filename"].unique():
            continue
        cl_grp = clusters_df[clusters_df["image_filename"] == img_fn].copy()
        gt_grp = gt_grp.sort_values("gcx")
        cl_grp = cl_grp.sort_values("center_x")
        n = min(len(gt_grp), len(cl_grp))
        if n == 0: continue
        gt_sel = gt_grp.head(n).reset_index(drop=True)
        cl_sel = cl_grp.head(n).reset_index(drop=True)
        for i in range(n):
            rows.append({
                "grid_path": cl_sel.loc[i, "grid_path"],
                "label": str(gt_sel.loc[i, "char_label"]).upper(),
                "gt_cx": gt_sel.loc[i, "gcx"],
                "gt_cy": gt_sel.loc[i, "gcy"],
                "gt_w":  gt_sel.loc[i, "gw"],
                "pred_cx": cl_sel.loc[i, "center_x"],
                "pred_cy": cl_sel.loc[i, "center_y"]
            })
    return pd.DataFrame(rows)

# =========================
# Dataset (그리드 이미지 사용)
# =========================
class GridDataset(Dataset):
    def __init__(self, df, input_size=28, analysis_dir=None):
        self.df = df.reset_index(drop=True)
        self.analysis_dir = analysis_dir
        self.t = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        grid_path = str(row["grid_path"])
        if not os.path.isabs(grid_path) and self.analysis_dir is not None:
            grid_path = os.path.join(self.analysis_dir, grid_path)
        if not os.path.exists(grid_path) and self.analysis_dir is not None:
            alt = os.path.join(self.analysis_dir, os.path.basename(grid_path))
            if os.path.exists(alt): grid_path = alt
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"Grid image not found: {grid_path}")
        img = Image.open(grid_path).convert("L")
        img = self.t(img)
        y = label_to_id(row["label"])
        return img, y

# =========================
# 간단 CNN
# =========================
class CNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)  # 14->7
        )
        self.fc = nn.Sequential(
            nn.Linear(32*7*7,128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================
# 평가 (라벨/위치)
# =========================
@torch.no_grad()
def evaluate_with_position(model, loader, device, pairs_df, idx_offset=0):
    model.eval()
    correct_label = 0
    correct_both  = 0
    total = 0
    ptr = idx_offset
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        pred = logits.argmax(1).cpu().numpy()
        lab  = labels.numpy()
        bs = len(lab)
        correct_label += (pred == lab).sum()
        for b in range(bs):
            row = pairs_df.iloc[ptr + b]
            dist = math.hypot(row["pred_cx"] - row["gt_cx"], row["pred_cy"] - row["gt_cy"])
            pos_ok = (dist <= row["gt_w"])
            if (pred[b] == lab[b]) and pos_ok:
                correct_both += 1
        total += bs
        ptr += bs
    return (correct_label/total if total else 0.0), (correct_both/total if total else 0.0)

# =========================
# 트레이닝 실행 로직 (스레드에서 호출)
# =========================
def run_training(params, log_fn):
    try:
        random.seed(params["seed"]); torch.manual_seed(params["seed"])
        os.makedirs(params["out_dir"], exist_ok=True)

        clusters_csv = os.path.join(params["analysis_dir"], "clusters.csv")
        if not os.path.exists(clusters_csv):
            raise FileNotFoundError(f"clusters.csv not found in analysis_dir: {clusters_csv}")

        log_fn(f"[INFO] chars_csv:    {params['chars_csv']}")
        log_fn(f"[INFO] analysis_dir: {params['analysis_dir']}")
        log_fn(f"[INFO] out_dir:      {params['out_dir']}")

        gt = load_chars_csv(params["chars_csv"])
        cl = load_clusters_csv(clusters_csv)
        pairs = pair_by_image_and_order(gt, cl)
        if pairs.empty:
            raise RuntimeError("매칭된 pair가 없습니다. clusters.csv와 chars.csv를 확인하세요.")

        n = len(pairs)
        n_val = max(1, int(n * params["val_ratio"]))
        train_df = pairs.iloc[:-n_val].reset_index(drop=True)
        val_df   = pairs.iloc[-n_val:].reset_index(drop=True)

        train_ds = GridDataset(train_df, params["input_size"], analysis_dir=params["analysis_dir"])
        val_ds   = GridDataset(val_df,   params["input_size"], analysis_dir=params["analysis_dir"])
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN(num_classes=len(VOCAB)).to(device)
        crit = nn.CrossEntropyLoss()
        opt  = optim.Adam(model.parameters(), lr=params["lr"])

        best_both = 0.0
        best_path = os.path.join(params["out_dir"], "dot_cnn_best.pth")
        last_path = os.path.join(params["out_dir"], "dot_cnn_last.pth")

        for ep in range(1, params["epochs"] + 1):
            # --- Train ---
            model.train()
            ep_loss = 0.0; ep_cnt = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = crit(out, labels)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += float(loss.item()) * imgs.size(0)
                ep_cnt += imgs.size(0)
            ep_loss /= max(1, ep_cnt)

            # --- Eval ---
            label_acc, both_acc = evaluate_with_position(model, val_loader, device, val_df, idx_offset=0)
            log_fn(f"[Epoch {ep:02d}] train_loss={ep_loss:.4f} | val_label_acc={label_acc*100:.2f}% | val_label+pos_acc={both_acc*100:.2f}%")

            # --- Save best ---
            if both_acc > best_both:
                best_both = both_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                    "input_size": params["input_size"]
                }, best_path)
                log_fn(f"  ↳ Saved BEST to {best_path} (label+pos_acc={best_both*100:.2f}%)")

            # --- Save last (매 epoch 덮어쓰기) ---
            if params["save_last"]:
                torch.save({
                    "model_state": model.state_dict(),
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                    "input_size": params["input_size"]
                }, last_path)

        log_fn(f"[DONE] Best (label+position) acc: {best_both*100:.2f}%")
        if os.path.exists(best_path): log_fn(f"Best model: {best_path}")
        if params["save_last"] and os.path.exists(last_path): log_fn(f"Last model: {last_path}")

    except Exception as e:
        log_fn(f"[ERROR] {e}")

# =========================
# Tkinter GUI (전체 스크롤 + 로그 스크롤)
# =========================
class TrainerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Dot Font Trainer (Grid-based)")

        # ---------- 전체 스크롤 컨테이너 ----------
        outer = ttk.Frame(master)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer, borderwidth=0)
        self.scrollbar_main = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.scrollbar_main.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar_main.set)

        # 스크롤 가능한 내부 프레임
        self.inner = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # 휠 스크롤
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # ---------- 폼: 경로/하이퍼파라미터 ----------
        pad = {"padx": 8, "pady": 4}

        row = 0
        ttk.Label(self.inner, text="chars.csv 경로").grid(row=row, column=0, sticky="w", **pad)
        self.var_chars = tk.StringVar(value="./manual_dot_image/chars.csv")
        ttk.Entry(self.inner, textvariable=self.var_chars, width=60).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self.inner, text="찾기", command=self.browse_chars).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(self.inner, text="analysis 결과 폴더").grid(row=row, column=0, sticky="w", **pad)
        self.var_analysis = tk.StringVar(value="./manual_dot_image/analysis_run1")
        ttk.Entry(self.inner, textvariable=self.var_analysis, width=60).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self.inner, text="찾기", command=self.browse_analysis).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(self.inner, text="모델 저장 폴더(out_dir)").grid(row=row, column=0, sticky="w", **pad)
        self.var_outdir = tk.StringVar(value="./models")
        ttk.Entry(self.inner, textvariable=self.var_outdir, width=60).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self.inner, text="찾기", command=self.browse_outdir).grid(row=row, column=2, **pad)

        # 하이퍼파라미터
        row += 1
        ttk.Label(self.inner, text="epochs").grid(row=row, column=0, sticky="w", **pad)
        self.var_epochs = tk.IntVar(value=10)
        ttk.Spinbox(self.inner, from_=1, to=500, textvariable=self.var_epochs, width=8).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(self.inner, text="batch_size").grid(row=row, column=0, sticky="w", **pad)
        self.var_batch = tk.IntVar(value=64)
        ttk.Spinbox(self.inner, from_=1, to=2048, increment=1, textvariable=self.var_batch, width=8).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(self.inner, text="learning rate").grid(row=row, column=0, sticky="w", **pad)
        self.var_lr = tk.DoubleVar(value=1e-3)
        ttk.Entry(self.inner, textvariable=self.var_lr, width=12).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(self.inner, text="input_size").grid(row=row, column=0, sticky="w", **pad)
        self.var_inp = tk.IntVar(value=28)
        ttk.Spinbox(self.inner, from_=8, to=128, textvariable=self.var_inp, width=8).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(self.inner, text="val_ratio").grid(row=row, column=0, sticky="w", **pad)
        self.var_val = tk.DoubleVar(value=0.1)
        ttk.Entry(self.inner, textvariable=self.var_val, width=12).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(self.inner, text="seed").grid(row=row, column=0, sticky="w", **pad)
        self.var_seed = tk.IntVar(value=42)
        ttk.Spinbox(self.inner, from_=0, to=999999, textvariable=self.var_seed, width=10).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        self.var_save_last = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.inner, text="마지막 에폭 모델 저장(dot_cnn_last.pth)", variable=self.var_save_last).grid(row=row, column=0, columnspan=2, sticky="w", **pad)

        # 실행 버튼
        row += 1
        self.btn_start = ttk.Button(self.inner, text="학습 시작", command=self.start_training)
        self.btn_start.grid(row=row, column=0, sticky="w", **pad)
        self.btn_stop  = ttk.Button(self.inner, text="중지", command=self.stop_training, state="disabled")
        self.btn_stop.grid(row=row, column=1, sticky="w", **pad)

        # 진행바
        row += 1
        ttk.Label(self.inner, text="진행 상황").grid(row=row, column=0, sticky="w", **pad)
        self.progress = ttk.Progressbar(self.inner, mode="indeterminate")
        self.progress.grid(row=row, column=1, columnspan=2, sticky="we", **pad)

        # ---------- 로그 영역 (전용 스크롤) ----------
        row += 1
        ttk.Label(self.inner, text="Logs").grid(row=row, column=0, sticky="w", **pad)
        self.log_frame = ttk.Frame(self.inner)
        self.log_frame.grid(row=row, column=1, columnspan=2, sticky="nsew", **pad)

        self.log_text = tk.Text(self.log_frame, height=18, wrap="word")
        self.log_scroll = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_scroll.pack(side="right", fill="y")

        # 행/열 확장
        for c in range(3):
            self.inner.grid_columnconfigure(c, weight=1)

        # 로그 큐 (스레드→UI)
        self.log_queue = queue.Queue()
        self.train_thread = None
        self.stop_flag = False
        self.master.after(100, self._poll_log_queue)

    # 스크롤 휠: 전체 옵션 프레임
    def _on_mousewheel(self, event):
        # Windows(+120/-120), macOS uses different units
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def browse_chars(self):
        path = filedialog.askopenfilename(title="chars.csv 선택", filetypes=[("CSV","*.csv"),("All","*.*")])
        if path: self.var_chars.set(path)

    def browse_analysis(self):
        path = filedialog.askdirectory(title="analysis 결과 폴더 선택")
        if path: self.var_analysis.set(path)

    def browse_outdir(self):
        path = filedialog.askdirectory(title="모델 저장 폴더 선택")
        if path: self.var_outdir.set(path)

    def log(self, msg):
        # 스레드세이프: 큐에 넣고 UI 쓰레드에서 소비
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.master.after(100, self._poll_log_queue)

    def start_training(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showinfo("알림", "이미 학습이 진행 중입니다.")
            return

        # 파라미터 수집
        params = {
            "chars_csv":  self.var_chars.get().strip(),
            "analysis_dir": self.var_analysis.get().strip(),
            "out_dir":    self.var_outdir.get().strip(),
            "epochs":     int(self.var_epochs.get()),
            "batch_size": int(self.var_batch.get()),
            "lr":         float(self.var_lr.get()),
            "input_size": int(self.var_inp.get()),
            "val_ratio":  float(self.var_val.get()),
            "seed":       int(self.var_seed.get()),
            "save_last":  bool(self.var_save_last.get())
        }

        # 기본 검사
        if not os.path.exists(params["chars_csv"]):
            messagebox.showerror("에러", f"chars_csv 없음: {params['chars_csv']}")
            return
        if not os.path.isdir(params["analysis_dir"]):
            messagebox.showerror("에러", f"analysis_dir 폴더 없음: {params['analysis_dir']}")
            return
        os.makedirs(params["out_dir"], exist_ok=True)

        self.stop_flag = False
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress.start(10)
        self.log_text.delete("1.0", "end")
        self.log("[INFO] 학습을 시작합니다...")

        # 스레드로 학습 실행
        def _runner():
            run_training(params, self.log)
            # 종료 처리
            self.log("[INFO] 학습 스레드 종료")
            self.master.after(0, self._on_training_done)

        self.train_thread = threading.Thread(target=_runner, daemon=True)
        self.train_thread.start()

    def _on_training_done(self):
        self.progress.stop()
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def stop_training(self):
        messagebox.showinfo("알림", "중지 기능은 안전한 학습 종료를 위해 구현 예정입니다.\n현재 에폭이 끝나면 종료되도록 개선 권장.")
        # PyTorch 학습을 강제 중단하려면 프로세스 관리가 필요합니다.
        # 여기서는 안내만 제공합니다.

# =========================
# 시작
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = TrainerGUI(root)
    root.geometry("900x700")
    root.mainloop()
