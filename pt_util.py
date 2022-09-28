import imageio
import numpy as np

def render_fonts_image(x, path, img_per_row, unit_scale=True):
    if unit_scale:
        # scale 0-1 matrix back to gray scale bitmaps
        bitmaps = (x * 255.).astype(dtype=np.int16) % 256
    else:
        bitmaps = x
    num_imgs, w, h = x.shape
    assert w == h
    side = int(w)
    width = img_per_row * w
    height = int(np.ceil(float(num_imgs) / img_per_row)) * h
    canvas = np.zeros(shape=(height, width), dtype=np.int16)
    # make the canvas all white
    canvas.fill(255)
    for idx, bm in enumerate(bitmaps):
        x = h * int(idx / img_per_row)
        y = w * int(idx % img_per_row)
        canvas[x: x + h, y: y + w] = bm
    canvas = canvas.astype(np.uint8)
    imageio.imwrite(path, canvas)
    return path