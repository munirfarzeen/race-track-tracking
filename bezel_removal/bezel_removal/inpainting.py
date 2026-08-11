import cv2
import numpy as np


class Inpainter:
    def __init__(self, backend):
        self.backend = backend
        self.lama = None

        if backend in {"auto", "lama"}:
            try:
                from simple_lama_inpainting import SimpleLama

                self.lama = SimpleLama()
                self.backend = "lama"
            except ImportError:
                if backend == "lama":
                    raise RuntimeError(
                        "LaMa requested, but simple_lama_inpainting is not installed."
                    )
                self.backend = "opencv"

    def inpaint(self, image, mask, radius):
        if not np.any(mask):
            return image.copy()

        if self.backend == "lama":
            from PIL import Image

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_mask = Image.fromarray(mask)
            result = self.lama(pil_image, pil_mask)
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def feathered_composite(original, inpainted, mask, feather):
    if feather <= 0 or not np.any(mask):
        return inpainted

    ksize = feather if feather % 2 == 1 else feather + 1
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    alpha = np.clip(alpha[:, :, None], 0.0, 1.0)
    blended = (
        original.astype(np.float32) * (1.0 - alpha)
        + inpainted.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)

