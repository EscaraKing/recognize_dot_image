# dot_font_trainer.py  (그리드 기반 학습/평가, 분석 하위폴더 지원 + 체크포인트 저장)
import os, math, argparse, random
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ===== 라벨 매핑 (0-9 + A-Z) =====
VOCAB = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z')+1)]
LABEL2ID = {ch:i for i,ch in enumerate(VOCAB)}
ID2LABEL = {i:ch for ch,i in LABEL2ID.items()}

def label_to_id(ch):
    ch = str(ch).upper()
    if ch not in LABEL2ID: raise ValueError(f"Unsupported label: {ch}")
    return LABEL2ID[ch]

# ===== CSV 로더 =====
def load_chars_csv(chars_csv):
    df = pd.read_csv(chars_csv)
    df["image_filename"] = df["image_path"].apply(lambda p: os.path.basename(str(p)))
    df["gcx"] = (df["char_box_x1"] + df["char_box_x2"]) / 2.0
    df["gcy"] = (df["char_box_y1"] + df["char_box_y2"]) / 2.0
    df["gw"]  = (df["char_box_x2"] - df["char_box_x1"]).clip(lower=1.0)
    # 지원 라벨만 사용
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
    """이미지별로 정렬 후, 같은 인덱스끼리 매칭 → grid_path ↔ (label, gt center/width)."""
    pairs = []
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
            pairs.append({
                "grid_path": cl_sel.loc[i, "grid_path"],
                "label": str(gt_sel.loc[i, "char_label"]).upper(),
                "gt_cx": gt_sel.loc[i, "gcx"],
                "gt_cy": gt_sel.loc[i, "gcy"],
                "gt_w":  gt_sel.loc[i, "gw"],
                "pred_cx": cl_sel.loc[i, "center_x"],
                "pred_cy": cl_sel.loc[i, "center_y"]
            })
    return pd.DataFrame(pairs)

# ===== Dataset (그리드 이미지 사용) =====
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

        # 절대경로가 아니면 analysis_dir 기준으로 보정
        if not os.path.isabs(grid_path):
            if self.analysis_dir is not None:
                grid_path = os.path.join(self.analysis_dir, grid_path)

        # 백업: 경로가 여전히 없으면 파일명으로 탐색
        if not os.path.exists(grid_path) and self.analysis_dir is not None:
            fallback = os.path.join(self.analysis_dir, os.path.basename(grid_path))
            if os.path.exists(fallback):
                grid_path = fallback

        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"Grid image not found: {grid_path}")

        img = Image.open(grid_path).convert("L")
        img = self.t(img)
        y = label_to_id(row["label"])
        return img, y

# ===== 간단 CNN =====
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

@torch.no_grad()
def evaluate_with_position(model, loader, device, pairs_df, idx_offset=0):
    """
    평가: (1) 라벨 정확도, (2) 라벨+위치 동시 정확도(중심 거리 ≤ GT 폭).
    DataLoader는 shuffle=False 이어야 인덱스 정렬이 유지됨.
    """
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
        # 라벨 정확도
        correct_label += (pred == lab).sum()
        # 위치 기준: 중심 거리 <= gt 폭
        for b in range(bs):
            row = pairs_df.iloc[ptr + b]
            dist = math.hypot(row["pred_cx"] - row["gt_cx"], row["pred_cy"] - row["gt_cy"])
            pos_ok = (dist <= row["gt_w"])
            if (pred[b] == lab[b]) and pos_ok:
                correct_both += 1
        total += bs
        ptr += bs
    return correct_label/total, correct_both/total

def main(
    chars_csv="./manual_dot_image/chars.csv",
    analysis_dir="./manual_dot_image/analysis_run1",
    out_dir="./models",
    epochs=10, batch_size=64, lr=1e-3, input_size=28, val_ratio=0.1, seed=42,
    save_last: bool = True
):
    random.seed(seed); torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    clusters_csv = os.path.join(analysis_dir, "clusters.csv")
    if not os.path.exists(clusters_csv):
        raise FileNotFoundError(f"clusters.csv not found in analysis_dir: {clusters_csv}")

    gt = load_chars_csv(chars_csv)
    cl = load_clusters_csv(clusters_csv)
    pairs = pair_by_image_and_order(gt, cl)
    if pairs.empty:
        raise RuntimeError("매칭된 pair가 없습니다. clusters.csv와 chars_csv를 확인하세요.")

    # 학습/검증 분할
    n = len(pairs)
    n_val = max(1, int(n*val_ratio))
    train_df = pairs.iloc[:-n_val].reset_index(drop=True)
    val_df   = pairs.iloc[-n_val:].reset_index(drop=True)

    train_ds = GridDataset(train_df, input_size, analysis_dir=analysis_dir)
    val_ds   = GridDataset(val_df,   input_size, analysis_dir=analysis_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(VOCAB)).to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=lr)

    best_both = 0.0
    best_path = os.path.join(out_dir, "dot_cnn_best.pth")
    last_path = os.path.join(out_dir, "dot_cnn_last.pth")

    for ep in range(1, epochs+1):
        # === train ===
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = crit(out, labels)
            opt.zero_grad(); loss.backward(); opt.step()

        # === eval (라벨/위치) ===
        label_acc, both_acc = evaluate_with_position(
            model, val_loader, device, val_df, idx_offset=0
        )
        print(f"[Epoch {ep:02d}] val label_acc={label_acc*100:.2f}% | label+pos_acc={both_acc*100:.2f}%")

        # === save best (기준: both_acc) ===
        if both_acc > best_both:
            best_both = both_acc
            torch.save({
                "model_state": model.state_dict(),
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "input_size": input_size
            }, best_path)
            print(f"  ↳ Saved BEST to {best_path} (label+pos_acc={best_both*100:.2f}%)")

        # === save last (옵션) ===
        if save_last:
            torch.save({
                "model_state": model.state_dict(),
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "input_size": input_size
            }, last_path)

    print(f"Best (label+position) acc: {best_both*100:.2f}%")
    if os.path.exists(best_path):
        print(f"Best model: {best_path}")
    if save_last and os.path.exists(last_path):
        print(f"Last model: {last_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chars_csv", type=str, default="./manual_dot_image/chars.csv")
    parser.add_argument("--analysis_dir", type=str, default="./manual_dot_image/analysis_run1")
    parser.add_argument("--out_dir", type=str, default="./models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input_size", type=int, default=28)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_last", action="store_true")  # 지정하면 마지막 에폭도 저장
    args = parser.parse_args()
    main(**vars(args))
