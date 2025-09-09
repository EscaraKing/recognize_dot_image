# dot_font_evaluator.py
import os, math, argparse
import pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ===== 공통: 라벨 매핑 (0-9 + A-Z) =====
VOCAB = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z')+1)]
LABEL2ID = {ch:i for i,ch in enumerate(VOCAB)}
ID2LABEL = {i:ch for ch,i in LABEL2ID.items()}

def label_to_id(ch):
    ch = str(ch).upper()
    if ch not in LABEL2ID: raise ValueError(f"Unsupported label: {ch}")
    return LABEL2ID[ch]

# ===== 모델 정의 (트레이너와 동일) =====
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

# ===== 데이터 적재 =====
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
    """
    이미지별로 x중심 오름차순 정렬 후 같은 인덱스끼리 매칭.
    반환 DF 컬럼:
      image_filename, cluster_index, grid_path, label,
      gt_cx, gt_cy, gt_w, pred_cx, pred_cy
    """
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
                "image_filename": img_fn,
                "cluster_index": int(cl_sel.loc[i, "cluster_index"]),
                "grid_path": cl_sel.loc[i, "grid_path"],
                "label": str(gt_sel.loc[i, "char_label"]).upper(),
                "gt_cx": gt_sel.loc[i, "gcx"],
                "gt_cy": gt_sel.loc[i, "gcy"],
                "gt_w":  gt_sel.loc[i, "gw"],
                "pred_cx": cl_sel.loc[i, "center_x"],
                "pred_cy": cl_sel.loc[i, "center_y"]
            })
    return pd.DataFrame(rows)

class GridEvalDataset(Dataset):
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
        # 상대경로 보정
        if not os.path.isabs(grid_path) and self.analysis_dir is not None:
            grid_path = os.path.join(self.analysis_dir, grid_path)
        # 파일명 fallback
        if not os.path.exists(grid_path) and self.analysis_dir is not None:
            alt = os.path.join(self.analysis_dir, os.path.basename(grid_path))
            if os.path.exists(alt): grid_path = alt
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"Grid image not found: {grid_path}")
        img = Image.open(grid_path).convert("L")
        img = self.t(img)
        y = label_to_id(row["label"])
        return img, y

# ===== 평가 =====
@torch.no_grad()
def run_eval(model, loader, device, pairs_df, idx_offset=0):
    """
    라벨 정확도, 라벨+위치 동시 정확도 계산과 예측 결과 수집.
    """
    model.eval()
    total = 0
    label_correct = 0
    both_correct = 0
    preds_rows = []

    ptr = idx_offset
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = logits.argmax(1).cpu().numpy()
        labs  = labels.numpy()
        bs = len(labs)

        for b in range(bs):
            row = pairs_df.iloc[ptr + b]
            pred_id = int(preds[b])
            lab_id  = int(labs[b])
            pred_ch = ID2LABEL.get(pred_id, "?")
            lab_ch  = ID2LABEL.get(lab_id, "?")
            # 위치 판정
            dist = math.hypot(row["pred_cx"] - row["gt_cx"], row["pred_cy"] - row["gt_cy"])
            pos_ok = (dist <= row["gt_w"])

            # 정확도 카운트
            label_ok = (pred_id == lab_id)
            if label_ok: label_correct += 1
            if label_ok and pos_ok: both_correct += 1
            total += 1

            # 예측 확률(최대값) 저장
            conf = float(probs[b, pred_id].item())

            preds_rows.append({
                "image_filename": row["image_filename"],
                "cluster_index": row["cluster_index"],
                "grid_path": row["grid_path"],
                "pred_label": pred_ch,
                "pred_conf": round(conf, 4),
                "gt_label": lab_ch,
                "center_dist": round(dist, 3),
                "gt_w": round(float(row["gt_w"]), 3),
                "position_ok": bool(pos_ok),
                "label_ok": bool(label_ok)
            })
        ptr += bs

    label_acc = label_correct / max(1, total)
    both_acc  = both_correct / max(1, total)
    return label_acc, both_acc, pd.DataFrame(preds_rows)

def classwise_accuracy(preds_df):
    stats = []
    for ch, grp in preds_df.groupby("gt_label"):
        n = len(grp)
        c1 = int((grp["label_ok"] == True).sum())
        c2 = int(((grp["label_ok"] == True) & (grp["position_ok"] == True)).sum())
        stats.append({
            "label": ch,
            "samples": n,
            "label_acc": round(c1 / n * 100, 2) if n else 0.0,
            "label_pos_acc": round(c2 / n * 100, 2) if n else 0.0
        })
    return pd.DataFrame(stats).sort_values("label")

def main(chars_csv, analysis_dir, ckpt, input_size=28, batch_size=128, num_workers=0, out_csv="predictions.csv"):
    clusters_csv = os.path.join(analysis_dir, "clusters.csv")
    if not os.path.exists(clusters_csv):
        raise FileNotFoundError(f"clusters.csv not found: {clusters_csv}")

    # 데이터 페어링
    gt = load_chars_csv(chars_csv)
    cl = load_clusters_csv(clusters_csv)
    pairs = pair_by_image_and_order(gt, cl)
    if pairs.empty:
        raise RuntimeError("매칭된 pair가 없습니다. clusters.csv와 chars.csv를 확인하세요.")

    # 데이터셋/로더
    ds = GridEvalDataset(pairs, input_size=input_size, analysis_dir=analysis_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_obj = torch.load(ckpt, map_location=device)
    # 체크포인트에 input_size가 있으면 따라감
    if isinstance(ckpt_obj, dict) and "input_size" in ckpt_obj:
        input_size = int(ckpt_obj["input_size"])

    model = CNN(num_classes=len(VOCAB)).to(device)
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        model.load_state_dict(ckpt_obj["model_state"])
    else:
        # 순수 state_dict만 저장된 경우
        model.load_state_dict(ckpt_obj)
    model.eval()

    # 평가
    label_acc, both_acc, preds_df = run_eval(model, loader, device, pairs)
    print(f"[EVAL] label_acc={label_acc*100:.2f}% | label+position_acc={both_acc*100:.2f}%")

    # 클래스별 리포트
    report_df = classwise_accuracy(preds_df)
    print("\nPer-class accuracy (label only / label+position):")
    print(report_df.to_string(index=False))

    # 결과 저장
    out_path = os.path.join(analysis_dir, out_csv)
    preds_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved predictions to: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars_csv", type=str, default="./manual_dot_image/chars.csv")
    ap.add_argument("--analysis_dir", type=str, default="./manual_dot_image/analysis_run1")
    ap.add_argument("--ckpt", type=str, default="./models/dot_cnn_best.pth")
    ap.add_argument("--input_size", type=int, default=28)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="predictions.csv")
    args = ap.parse_args()
    main(**vars(args))
