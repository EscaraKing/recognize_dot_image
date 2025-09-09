from PIL import Image, ImageDraw

# Example: 5x7 dot font, update or expand for more/less dots per character
DOT_FONT_5x7 = {
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001"
    ],
    "Y": [
        "10001",
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "00100"
    ],
    "0": [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110"
    ],
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110"
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111"
    ],
    "3": [
        "01110",
        "10001",
        "00001",
        "00110",
        "00001",
        "10001",
        "01110"
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010"
    ],
    "5": [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110"
    ],
    "6": [
        "01110",
        "10000",
        "11110",
        "10001",
        "10001",
        "10001",
        "01110"
    ],
    "7": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000"
    ],
    "8": [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110"
    ],
    "9": [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00001",
        "01110"
    ],
    " ": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000"
    ]
}

DOT_FONT_7x9 = {
    "A": [
        "0011100",
        "0100010",
        "1000001",
        "1000001",
        "1111111",
        "1000001",
        "1000001",
        "1000001",
        "1000001"
    ],
    "Y": [
        "1000001",
        "1000001",
        "0100010",
        "0010100",
        "0001000",
        "0001000",
        "0001000",
        "0001000",
        "0001000"
    ],
    "0": [
        "0011100",
        "0100010",
        "1001001",
        "1010101",
        "1010101",
        "1001001",
        "0100010",
        "0011100",
        "0000000"
    ],
    "1": [
        "0001000",
        "0011000",
        "0001000",
        "0001000",
        "0001000",
        "0001000",
        "0001000",
        "0011100",
        "0000000"
    ],
    "2": [
        "0011100",
        "0100010",
        "0000010",
        "0000100",
        "0001000",
        "0010000",
        "0100000",
        "1111110",
        "0000000"
    ],
    "3": [
        "0011100",
        "0100010",
        "0000010",
        "0001100",
        "0000010",
        "0000010",
        "0100010",
        "0011100",
        "0000000"
    ],
    "4": [
        "0000100",
        "0001100",
        "0010100",
        "0100100",
        "1111110",
        "0000100",
        "0000100",
        "0000100",
        "0000000"
    ],
    "5": [
        "1111110",
        "1000000",
        "1111100",
        "0000010",
        "0000010",
        "0000010",
        "1000010",
        "0111100",
        "0000000"
    ],
    "6": [
        "0011100",
        "0100000",
        "1000000",
        "1111100",
        "1000010",
        "1000010",
        "1000010",
        "0111100",
        "0000000"
    ],
    "7": [
        "1111110",
        "0000010",
        "0000100",
        "0001000",
        "0010000",
        "0100000",
        "0100000",
        "0100000",
        "0000000"
    ],
    "8": [
        "0111100",
        "1000010",
        "1000010",
        "0111100",
        "1000010",
        "1000010",
        "1000010",
        "0111100",
        "0000000"
    ],
    "9": [
        "0111100",
        "1000010",
        "1000010",
        "0111110",
        "0000010",
        "0000010",
        "0000010",
        "0111100",
        "0000000"
    ],
    " ": [
        "0000000",
        "0000000",
        "0000000",
        "0000000",
        "0000000",
        "0000000",
        "0000000",
        "0000000",
        "0000000"
    ]
}

def render_dot_text(
    text,
    font=DOT_FONT_5x7,
    dot_size=20,
    dot_spacing=10,
    line_spacing=20,
    fg_color=(0,0,0),
    bg_color=(255,255,255)
):
    text = text.upper()
    rows = len(next(iter(font.values())))
    cols = len(font[next(iter(font))][0])
    img_width = (cols * dot_spacing + dot_spacing) * len(text) + dot_spacing
    img_height = rows * dot_spacing + line_spacing
    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    for idx, char in enumerate(text):
        grid = font.get(char, font[" "])
        for y in range(rows):
            for x in range(cols):
                if grid[y][x] == "1":
                    cx = idx * (cols * dot_spacing + dot_spacing) + x * dot_spacing + dot_spacing
                    cy = y * dot_spacing + dot_spacing
                    draw.ellipse(
                        [cx-dot_size//2, cy-dot_size//2, cx+dot_size//2, cy+dot_size//2],
                        fill=fg_color
                    )
    return img

if __name__ == "__main__":
    import random
    import string

    # Parameters
    import os
    num_images = 10  # Number of images to generate
    letter_length = 6  # Length of random letter string in each image
    dot_size = 5         # Change dot size here
    dot_spacing = 40     # Change spacing between dots here
    output_dir = "./recognize_dot_image/images/"  # Change this to your desired output path
    os.makedirs(output_dir, exist_ok=True)

    # Select font grid: '5x7' or '7x9'
    font_choice = "5x7"  # Change to "7x9" for larger grid
    if font_choice == "5x7":
        font_grid = DOT_FONT_5x7
    elif font_choice == "7x9":
        font_grid = DOT_FONT_7x9
    else:
        raise ValueError("Unknown font grid choice")

    # Allowed characters
    allowed_chars = list(font_grid.keys())
    allowed_chars.remove(" ")  # Exclude space from random generation

    for i in range(num_images):
        text = "".join(random.choices(allowed_chars, k=letter_length))
        img = render_dot_text(
            text,
            font=font_grid,
            dot_size=dot_size,
            dot_spacing=dot_spacing,
            fg_color=(60, 120, 255),
            bg_color=(255,255,255)
        )
        img.save(os.path.join(output_dir, f"dot_font_random_{i+1}.png"))