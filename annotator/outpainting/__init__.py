'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Ning Yu
'''

import numpy as np

class Outpainter:
    def __call__(self, img, height_top_extended, height_down_extended, width_left_extended, width_right_extended):
        height, width, channel = img.shape

        height_top_new = int(float(height) / 100.0 * float(height_top_extended))
        height_down_new = int(float(height) / 100.0 * float(height_down_extended))
        width_left_new = int(float(width) / 100.0 * float(width_left_extended))
        width_right_new = int(float(width) / 100.0 * float(width_right_extended))

        new_height = height + height_top_new + height_down_new
        new_width = width + width_left_new + width_right_new
        img_new = np.zeros([new_height, new_width, channel])
        img_new[height_top_new: (height + height_top_new), width_left_new: (width + width_left_new), : ] = img
        img_new = img_new.astype('ubyte')
        return img_new
