#!/usr/bin/env python3

import pathlib
import sys


OLD_IMPORT_BLOCK = """import time
from io import BytesIO
import cv2
import ctypes
from wand.api import library as wandlibrary
from wand.image import Image as WandImage
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom

# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
"""


NEW_IMPORT_BLOCK = """import time
from io import BytesIO
import cv2
import ctypes
try:
    from wand.api import library as wandlibrary
    from wand.image import Image as WandImage
except Exception:
    wandlibrary = None
    WandImage = None
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom

if wandlibrary is not None:
    wandlibrary.MagickMotionBlurImage.argtypes = (
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    )


class MotionImage(WandImage if WandImage is not None else object):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        if wandlibrary is None:
            raise RuntimeError("MagickWand is unavailable")
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
"""


OLD_MOTION_BLUR_BLOCK = """def motion_blur(x, severity=1):
    # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    # c = [(15, 5), (20, 8), (25, 12), (30, 15), (35, 20)][severity - 1]
    c = [
        (5, 2),
        (8, 3),
        (10, 4),
        (12, 5),
        (15, 6),
        (18, 8),
        (20, 10),
        (25, 12),
        (30, 15),
        (35, 20)
    ][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                    cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
"""


NEW_MOTION_BLUR_BLOCK = """def motion_blur(x, severity=1):
    # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    # c = [(15, 5), (20, 8), (25, 12), (30, 15), (35, 20)][severity - 1]
    c = [
        (5, 2),
        (8, 3),
        (10, 4),
        (12, 5),
        (15, 6),
        (18, 8),
        (20, 10),
        (25, 12),
        (30, 15),
        (35, 20)
    ][severity - 1]

    if wandlibrary is not None and WandImage is not None:
        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())
        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        if x.shape != (224, 224):
            return np.clip(x[..., [2, 1, 0]], 0, 255)
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

    img = np.array(x)
    ksize = int(c[0])
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    center = (ksize / 2 - 0.5, ksize / 2 - 0.5)
    angle = float(np.random.uniform(-45, 45))
    rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotate, (ksize, ksize))
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0:
        kernel /= kernel_sum
    blurred = cv2.filter2D(img, -1, kernel)
    return np.clip(blurred, 0, 255)
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_libero_plus_env_wrapper.py /path/to/env_wrapper.py", file=sys.stderr)
        return 2

    path = pathlib.Path(sys.argv[1])
    text = path.read_text()

    if "MagickWand is unavailable" in text:
        print(f"Already patched: {path}")
        return 0

    if OLD_IMPORT_BLOCK not in text:
        print(f"Did not find expected import block in {path}", file=sys.stderr)
        return 1
    if OLD_MOTION_BLUR_BLOCK not in text:
        print(f"Did not find expected motion blur block in {path}", file=sys.stderr)
        return 1

    text = text.replace(OLD_IMPORT_BLOCK, NEW_IMPORT_BLOCK)
    text = text.replace(OLD_MOTION_BLUR_BLOCK, NEW_MOTION_BLUR_BLOCK)
    path.write_text(text)
    print(f"Patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
