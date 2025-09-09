#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_exchange.py
- YOLO 포맷(이미지 + 라벨 txt)을 우리 파이프라인(manual_dot_image) 형식으로 변환
- 출력: manual_dot_image/images/ + manual_dot_image/chars.csv

YOLO 라벨 포맷 (한 줄당):
  <class_id> <cx> <cy> <w> <h>
  (cx,cy,w,h 는 0~1 정규화 좌표)

classes.txt(선택):
  각 줄에 클래스 이름(라벨) 하나씩.
  없으면 기본 매핑: ["0","1",...,"9","A",...,"Z"]

사용 예:
  python yolo_exchange.py \
    --yolo_dir ./yolo_image \
    --out_dir ./manual_dot_image \
    --copy_mode copy   # copy|link|move
"""

import os
import csv
import glob
import shutil
import argparse
from typing import List, Tuple
from PIL import Image

DEFAULT_VOCAB = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z')+1)]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def load_classes(classes_txt: str) -> List[str]:
    if classes_txt and os.path.exists(classes_txt):
        with open(classes_txt, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip() != ""]
        if names:
            return names
    # fallback
    return DEFAULT_VOCAB

def yolo_txt_to_boxes(txt_path: str, W: int, H: int) -> List[Tuple[int,int,int,int,int]]:
    """
    YOLO txt -> [(class_id, x1,y1,x2,y2), ...] (pixel coords, int)
    """
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 5:
                continue
            try:
                cid = int(float(sp[0]))
                cx = float(sp[1]) * W
                cy = float(sp[2]) * H
                w  = float(sp[3]) * W
                h  = float(sp[4]) * H
            except Exception:
                continue
            x1 = int(round(cx - w/2)); y1 = int(round(cy - h/2))
            x2 = int(round(cx + w/2)); y2 = int(round(cy + h/2))
            x1 = max(0, min(W-1, x1))
            y1 = max(0, min(H-1, y1))
            x2 = max(0, min(W-1, x2))
            y2 = max(0, min(H-1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((cid, x1, y1, x2, y2))
    return boxes

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def resolve_copy(src: str, dst: str, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "link":
        # 심볼릭 링크 (권한/OS에 따라 실패할 수 있음 → 그때는 copy로 폴백 권장)
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    else:
        raise ValueError(f"unknown copy_mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_dir", type=str, default="./yolo_image", help="YOLO 이미지/라벨 폴더")
    ap.add_argument("--out_dir",  type=str, default="./manual_dot_image", help="출력 루트 폴더")
    ap.add_argument("--classes",  type=str, default=None, help="classes.txt 경로 (없으면 기본 0-9,A-Z)")
    ap.add_argument("--image_subdir", type=str, default="images", help="이미지 저장 하위 폴더명")
    ap.add_argument("--copy_mode", type=str, default="copy", choices=["copy","link","move"], help="이미지 이동 방식")
    ap.add_argument("--chars_csv_name", type=str, default="chars.csv", help="출력 CSV 파일명")
    args = ap.parse_args()

    yolo_dir = args.yolo_dir
    out_dir  = args.out_dir
    classes_txt = args.classes or os.path.join(yolo_dir, "classes.txt")
    image_out_dir = os.path.join(out_dir, args.image_subdir)
    chars_csv_path = os.path.join(out_dir, args.chars_csv_name)

    if not os.path.isdir(yolo_dir):
        raise FileNotFoundError(f"YOLO 폴더가 없어요: {yolo_dir}")

    # 클래스 로드
    class_names = load_classes(classes_txt)
    print(f"[INFO] classes: {class_names}")

    # 출력 디렉토리 준비
    ensure_dir(out_dir)
    ensure_dir(image_out_dir)

    # 이미지 목록
    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(os.path.join(yolo_dir, f"*{ext}")))
    img_paths.sort()
    if not img_paths:
        print("[WARN] 이미지가 없습니다.")
        return

    rows = []
    skipped_images = 0
    total_boxes = 0

    for src_img in img_paths:
        base = os.path.splitext(os.path.basename(src_img))[0]
        txt_path = os.path.join(yolo_dir, base + ".txt")

        # 이미지 크기 읽기
        try:
            with Image.open(src_img) as im:
                W, H = im.size
        except Exception as e:
            print(f"[WARN] 이미지 열기 실패: {src_img} ({e})")
            continue

        boxes = yolo_txt_to_boxes(txt_path, W, H)
        if not boxes:
            # 라벨 없으면 스킵(원하면 이미지만 복사하도록 바꿀 수 있음)
            skipped_images += 1
            continue

        # 이미지 복사/링크/이동 → manual_dot_image/images/
        dst_img = os.path.join(image_out_dir, os.path.basename(src_img))
        if not os.path.exists(dst_img) or os.path.abspath(dst_img) == os.path.abspath(src_img) and args.copy_mode != "copy":
            # 동일 디스크/경로 보호. 필요시 조건 조정
            resolve_copy(src_img, dst_img, args.copy_mode)

        # CSV용 image_path는 절대경로보다 프로젝트 상대경로가 편리
        image_path_for_csv = os.path.relpath(dst_img, start=".")

        # 각 바운딩박스를 chars.csv 한 줄로
        for (cid, x1, y1, x2, y2) in boxes:
            # 클래스 이름 → 라벨(문자)
            label = class_names[cid] if 0 <= cid < len(class_names) else "?"
            label = str(label).upper()

            rows.append({
                "image_path": image_path_for_csv,
                "char_label": label,
                "char_box_x1": int(x1),
                "char_box_y1": int(y1),
                "char_box_x2": int(x2),
                "char_box_y2": int(y2),
                # 참고용(트레이너는 사용 안하지만 보존해도 무방)
                "image_width":  int(W),
                "image_height": int(H),
                "class_id":     int(cid),
                "source":       "yolo_exchange"
            })
            total_boxes += 1

    if not rows:
        print("[WARN] 변환할 라벨이 없습니다. (YOLO txt 비어있음)")
        return

    # chars.csv 저장
    header = [
        "image_path","char_label",
        "char_box_x1","char_box_y1","char_box_x2","char_box_y2",
        "image_width","image_height","class_id","source"
    ]
    with open(chars_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] {chars_csv_path} 저장 완료")
    print(f"       이미지 변환: {len(set([r['image_path'] for r in rows]))}장")
    print(f"       라벨 박스  : {total_boxes}개")
    if skipped_images:
        print(f"[NOTE] 라벨(.txt) 없어서 스킵한 이미지: {skipped_images}장")
    print(f"[HINT] 이제 dot_font_analyzer 로 분석(클러스터/그리드 생성) → dot_font_trainer 로 학습하세요.")
    
if __name__ == "__main__":
    main()
