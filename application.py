import sys

import cv2
import numpy as np
import torch
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from torchvision.transforms import v2

from ui.main_window import Ui_MainWindow
from model import build


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        # 将hex列表中所有hex格式(十六进制)的颜色转换rgb格式的颜色
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        # 颜色个数
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        # 根据输入的index 选择对应的rgb颜色
        c = self.palette[int(i) % self.n]
        # 返回选择的颜色 默认是rgb
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        # hex -> rgb
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # load the UI file
        self.setWindowTitle('Demo')
        self.model = None
        self.checkpoint = None
        self.img = None
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        # build signal slot
        self.btn_model.clicked.connect(self.load_model)
        self.btn_img.clicked.connect(self.load_img)
        self.btn_begin.clicked.connect(self.begin)

    def load_model(self):
        path, ok = QFileDialog.getOpenFileName(self, "选择模型", "./", "Checkpoint Files (*.pth)")
        if ok:
            self.label_model.setText(path)
            self.checkpoint = torch.load(path, map_location='cpu')
            self.model = build(self.checkpoint['opts'])
            self.model.load_state_dict(self.checkpoint['model'])
            self.model.cuda()  # TODO: use cuda?
            self.model.eval()

    def load_img(self):
        path, ok = QFileDialog.getOpenFileName(self, "选择图片", "./", "Image Files (*.jpg *.png)")
        if ok:
            self.label_img.setText(path)
            self.img = cv2.imread(path)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            pixmap = QPixmap(path)
            self.label_in.setPixmap(pixmap)

    def begin(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
        elif self.img is None:
            QMessageBox.warning(self, "警告", "请先选择图片文件")
        else:
            # recognize the image by the model
            with torch.no_grad():
                device = next(self.model.parameters()).device
                prediction = self.model([self.transform(self.img).to(device)])[0]
            img_result = self.img.copy()
            self.plot(img_result, prediction, self.checkpoint['id2label'])
            h, w, c = img_result.shape
            bytes_per_line = c * w
            img_result = QImage(img_result.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img_result)
            self.label_in.setPixmap(pixmap)

    @staticmethod
    def plot(img, target, id2label, line_thickness=2):
        """
        Plot the bounding boxes on the image.
        :param img: The image. (ndarray)
        :param target: The targets. Contains the boxes([x1, y1, x2, y2]), masks, labels and scores.
        :param id2label: The mapping from id to label.
        :param line_thickness: The thickness of the bounding box.
        """
        assert img.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'

        colors = Colors()
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # label font thickness
        target_num = target['labels'].shape[0]  # the number of targets in the image
        for i in range(target_num):
            box = target['boxes'][i].tolist()
            label = target['labels'][i].item()
            score = target['scores'][i].item()
            color = colors(label)
            text = "{} {:.2f}".format(id2label[label], score)
            if score < 0.5:  # TODO: ignore low score
                continue

            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))  # (x1, y1), (x2, y2)
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # box
            t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]  # text size
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # text box
            # text
            cv2.putText(img, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            if "masks" in target:
                soft_mask = target['masks'][i]
                mask = torch.zeros_like(soft_mask, dtype=torch.uint8, device="cpu")
                mask[soft_mask > 0.5] = 1
                mask = mask.numpy()[0]
                alpha = 0.3  # transparency
                color_mask = np.zeros_like(img)
                for k in range(3):
                    color_mask[:, :, k][mask > 0] = color[k]
                color_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_RGB2RGBA)
                color_mask[:, :, 3] = (mask * alpha * 255).astype(np.uint8)
                color_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_RGBA2RGB)
                cv2.addWeighted(img, 1, color_mask, 0.5, 0, dst=img)  # mask


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
