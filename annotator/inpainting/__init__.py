import numpy as np

class Inpainter:
    def __call__(self, img, height_top_mask, height_down_mask, width_left_mask, width_right_mask):
        h = img.shape[0]
        w = img.shape[1]
        h_top_mask = int(float(h) / 100.0 * float(height_top_mask))
        h_down_mask = int(float(h) / 100.0 * float(height_down_mask))

        w_left_mask = int(float(w) / 100.0 * float(width_left_mask))
        w_right_mask = int(float(w) / 100.0 * float(width_right_mask))

        img_new = img
        img_new[h_top_mask:h_down_mask, w_left_mask:w_right_mask] = 0
        img_new = img_new.astype('ubyte')
        return img_new
