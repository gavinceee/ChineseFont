
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import imageio

FLAGS = None


def draw_char_bitmap(ch, font, char_size, x_offset = 0, y_offset=0):
    """将汉字ch以字体font画到位图上，字体大小为char_size，后两个参数为左上角的偏移量
    """
    image = Image.new("RGB", (char_size, char_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    gray = image.convert('L')
    return gray

