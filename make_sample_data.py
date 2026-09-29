"""Generate sample images so the lab runs offline.

* sample_images/  : three real photographs of DIFFERENT sizes (from scikit-image)
* data/class_a/   : 10 synthetic 'circle' images   (label 0)
* data/class_b/   : 10 synthetic 'stripes' images  (label 1)

Replace data/class_a and data/class_b with your own photographs if you wish;
the rest of the pipeline is unchanged.
"""
import numpy as np
from PIL import Image, ImageDraw
from skimage import data

rng = np.random.default_rng(42)

# ---- three real photos, three different shapes ---------------------------
Image.fromarray(data.astronaut()).save("sample_images/photo.jpg")   # 512x512
Image.fromarray(data.chelsea()).save("sample_images/cat.jpg")       # 300x451
Image.fromarray(data.coffee()).save("sample_images/coffee.jpg")     # 400x600

# ---- 20-image, 2-class dataset (varying sizes and colours) ---------------
def rand_colour():
    return tuple(int(v) for v in rng.integers(0, 256, 3))

for i in range(10):
    w, h = (int(v) for v in rng.integers(120, 320, 2))
    img = Image.new("RGB", (w, h), rand_colour())
    d = ImageDraw.Draw(img)
    r = int(rng.integers(min(w, h) // 5, min(w, h) // 2 - 2))
    cx, cy = w // 2 + int(rng.integers(-10, 10)), h // 2 + int(rng.integers(-10, 10))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rand_colour())
    img.save(f"data/class_a/circle_{i:02d}.png")

for i in range(10):
    w, h = (int(v) for v in rng.integers(120, 320, 2))
    img = Image.new("RGB", (w, h), rand_colour())
    d = ImageDraw.Draw(img)
    step = int(rng.integers(12, 30))
    c2 = rand_colour()
    for x in range(0, w, 2 * step):
        d.rectangle([x, 0, x + step, h], fill=c2)
    img.save(f"data/class_b/stripes_{i:02d}.png")
print("done")
