import os
import csv
import json
import random
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

# ------------------------ Dot Fonts ------------------------
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

# ------------------------ Renderer ------------------------
def render_dot_text(
    text: str,
    font: Dict[str, List[str]],
    dot_size: int = 20,
    dot_spacing: int = 40,
    line_spacing: int = 20,
    fg_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    return_metadata: bool = True
):
    text = text.upper()
    rows = len(next(iter(font.values())))
    cols = len(font[next(iter(font))][0])

    char_block_w = (cols * dot_spacing + dot_spacing)
    img_width = char_block_w * len(text) + dot_spacing
    img_height = rows * dot_spacing + line_spacing

    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    char_boxes = []  # 문자별 바운딩박스
    char_points = [] # 문자별 점 좌표

    for idx, ch in enumerate(text):
        grid = font.get(ch, font[" "])
        char_x0 = idx * char_block_w + dot_spacing
        char_y0 = dot_spacing
        char_x1 = char_x0 + cols * dot_spacing
        char_y1 = char_y0 + rows * dot_spacing
        char_boxes.append((char_x0, char_y0, char_x1, char_y1))

        pts_this_char = []
        for y in range(rows):
            for x in range(cols):
                if grid[y][x] == "1":
                    cx = idx * char_block_w + x * dot_spacing + dot_spacing
                    cy = y * dot_spacing + dot_spacing
                    draw.ellipse(
                        [cx - dot_size // 2, cy - dot_size // 2,
                         cx + dot_size // 2, cy + dot_size // 2],
                        fill=fg_color
                    )
                    pts_this_char.append((int(cx), int(cy)))
        char_points.append(pts_this_char)

    if return_metadata:
        meta = {"rows": rows, "cols": cols,
                "char_boxes": char_boxes, "char_points": char_points}
        return img, meta
    return img

# ------------------------ Dataset Builder ------------------------
def build_dataset_and_csv(
    output_dir: str,
    num_images: int = 10,
    letter_length: int = 6,
    font_choice: str = "5x7",
    dot_size: int = 5,
    dot_spacing: int = 40,
    line_spacing: int = 20,
    fg_color=(60, 120, 255),
    bg_color=(255, 255, 255),
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    images_csv = os.path.join(output_dir, "images.csv")
    chars_csv = os.path.join(output_dir, "chars.csv")
    imgs_dir = os.path.join(output_dir, "images")
    os.makedirs(imgs_dir, exist_ok=True)

    font_grid = DOT_FONT_5x7 if font_choice == "5x7" else DOT_FONT_7x9
    allowed_chars = list(font_grid.keys())
    if " " in allowed_chars:
        allowed_chars.remove(" ")

    random.seed(seed)

    with open(images_csv, "w", newline="", encoding="utf-8") as f_img, \
         open(chars_csv,  "w", newline="", encoding="utf-8") as f_chr:

        img_writer = csv.writer(f_img)
        chr_writer = csv.writer(f_chr)

        img_writer.writerow([
            "image_path","label","font","rows","cols",
            "dot_size","dot_spacing","line_spacing",
            "fg_color","bg_color","seed"
        ])
        chr_writer.writerow([
            "image_path","img_index","char_index","char_label",
            "char_box_x1","char_box_y1","char_box_x2","char_box_y2",
            "points_json"
        ])

        for i in range(num_images):
            label = "".join(random.choices(allowed_chars, k=letter_length))
            img, meta = render_dot_text(
                label, font=font_grid,
                dot_size=dot_size, dot_spacing=dot_spacing,
                line_spacing=line_spacing,
                fg_color=fg_color, bg_color=bg_color,
                return_metadata=True
            )

            img_path = os.path.join(imgs_dir, f"dot_font_random_{i+1}.png")
            img.save(img_path)

            # 이미지 단위 row
            img_writer.writerow([
                img_path, label, font_choice, meta["rows"], meta["cols"],
                dot_size, dot_spacing, line_spacing,
                fg_color, bg_color, seed
            ])

            # 문자 단위 rows
            for ci, ch in enumerate(label):
                x1,y1,x2,y2 = meta["char_boxes"][ci]
                pts = meta["char_points"][ci]
                chr_writer.writerow([
                    img_path, i, ci, ch,
                    x1,y1,x2,y2,
                    json.dumps(pts)
                ])

if __name__ == "__main__":
    build_dataset_and_csv(
        output_dir="./manual_dot_image/",
        num_images=100,      # 생성할 이미지 개수
        letter_length=6,    # 문자열 길이
        font_choice="5x7",  # "5x7" 또는 "7x9"
        dot_size=5,
        dot_spacing=40,
        line_spacing=20,
        fg_color=(60,120,255),
        bg_color=(255,255,255),
        seed=42
    )
