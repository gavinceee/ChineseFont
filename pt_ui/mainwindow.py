# -*- coding:utf-8 -*-
from ui_main import Ui_MainWindow
from pt_font_util import draw_char_bitmap

# import network
from pt_network import RewriteNet
import torch
import torchvision.transforms as transforms

from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot

from PIL import ImageQt
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image

import numpy as np


import sys


class MyForm(QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super(MyForm, self).__init__()
        self.setupUi(self)

        self.font_path = None
        self.mfont = None
        self.gray = None

        self.char_size = 64

        #TODO: init netwrok and load params
        self.model = RewriteNet(mode='small')
        self.model.load_state_dict(torch.load(r"D:\projects\gao_rewrite\writing.pth"))
        self.model.eval()

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([160, 160])
            ]
        )
        self.toPIL = transforms.ToPILImage()

    @pyqtSlot(bool)
    def on_selectFont_pushButton_clicked(self, state):
        self.font_path, _ = QFileDialog.getOpenFileName(self, "OpenFont", ".", "Image Files(*.ttf)")
        self.fontPath_label.setText(self.font_path)
        self.mfont = ImageFont.truetype(self.font_path, self.char_size)

    @pyqtSlot(bool)
    def on_generate_pushButton_clicked(self, state):
        the_word = self.word_lineEdit.text()
        self.gray = draw_char_bitmap(the_word, self.mfont, self.char_size)
        self.origin_label.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(self.gray)))

    @pyqtSlot(bool)
    def on_tranform_pushButton_clicked(self, state):
        # generator image from origin image and show in self.target_label
        img_tensor = self.image2tensor(self.gray)
        result = self.model(img_tensor)
        img_generated = result[0] # Why?
        a = self.toPIL((img_generated))
        a.save("test.png")
        self.target_label.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(self.toPIL((img_generated * 255).astype(dtype=np.int8)))))

    def image2tensor(self, image):
        return torch.unsqueeze(self.image_transform(image), dim=0) # Why?

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MyForm()
    form.show()
    sys.exit(app.exec_())
