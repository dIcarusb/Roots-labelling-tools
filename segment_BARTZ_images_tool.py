import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen
from PyQt5.QtWidgets import QMenuBar, QAction, QUndoStack
import cv2
from skimage import morphology
import math
import numpy as np
import sys


def scaled(d):
    return np.divide(d-d.min(), d.max()-d.min(), dtype="float32")

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]


        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


n=5
class ImageLoader(QtWidgets.QWidget): #QtWidgets.QWidget
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        #super().__init__()
        self.setWindowTitle("Segment-tool")
        self.setWindowIcon(QtGui.QIcon('VSI.gif'))
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
        layout = QtWidgets.QGridLayout(self)

        self.label = QtWidgets.QLabel()
        layout.addWidget(self.label, 0, 0, 1, 6)
        self.label.setMinimumSize(196*n, 148*n)
        # the label alignment property is always maintained even when the contents
        # change, so there is no need to set it each time
        #self.label.setAlignment(QtCore.Qt.AlignCenter)


        #self.loadImageButton = QtWidgets.QPushButton('Load image')
        #layout.addWidget(self.loadImageButton, 1, 0)

        self.nextImageButton = QtWidgets.QPushButton('Next image')
        layout.addWidget(self.nextImageButton)

        #self.maskImageButton = QtWidgets.QPushButton('Predict Mask')
        #layout.addWidget(self.maskImageButton)

        self.maskClearButton = QtWidgets.QPushButton('Clear')
        layout.addWidget(self.maskClearButton)

        self.undoButton = QtWidgets.QPushButton('Undo')
        layout.addWidget(self.undoButton)

        self.maskSaveButton = QtWidgets.QPushButton('Save mask')
        layout.addWidget(self.maskSaveButton)

        self.lenImageButton = QtWidgets.QPushButton('Predict T.L.')
        layout.addWidget(self.lenImageButton)

        self.length = QtWidgets.QLCDNumber()
        layout.addWidget(self.length)

        #self.loadImageButton.clicked.connect(self.loadImage)
        self.nextImageButton.clicked.connect(self.nextImage)
        #self.maskImageButton.clicked.connect(self.mask)
        self.lenImageButton.clicked.connect(self.update_length)
        self.maskClearButton.clicked.connect(self.clear)
        self.undoButton.clicked.connect(self.undo)
        self.maskSaveButton.clicked.connect(self.save_mask)

        self.count = 0
        self.dirIterator = None
        self.fileList = []
        self.filename = None
        image = QImage(self.size(), QImage.Format_RGB32)
        image.fill(Qt.white)
        self.image = QtGui.QPixmap(image)#.scaled(self.label.size(),
                #QtCore.Qt.KeepAspectRatio)

        self.undoStack = QUndoStack(self)
        self.overlap_mask=0
        self.drawing = False
        self.brushSize = 3
        self.brushColor = Qt.green # from here
        self.lastPoint = QPoint(self.label.x(), self.label.y())
        self.lines=[]

        #self.line = QtCore.QLine()

        self.mainMenu = QMenuBar(self)#self.menuBar()#QMenuBar()
        #fileMenu = self.mainMenu.addMenu("File")
        self.brushsize = self.mainMenu.addMenu("Brush Size")
        self.correct = self.mainMenu.addAction("Auto-Correct color")
        #brushColor = self.mainMenu.addMenu("Brush Color")
        self.correct.triggered.connect(self.autocorrect)

        twpxAction = QAction(QIcon("icons/twpx.png"), "20px", self)
        self.brushsize.addAction(twpxAction)
        twpxAction.triggered.connect(self.twPixel)

        threepxAction = QAction(QIcon("icons/threepx.png"), "30px", self)
        self.brushsize.addAction(threepxAction)
        threepxAction.triggered.connect(self.threePixel)

        sixpxAction = QAction(QIcon("icons/sixpx.png"), "60px", self)
        self.brushsize.addAction(sixpxAction)
        sixpxAction.triggered.connect(self.sixPixel)

        ninepxAction = QAction(QIcon("icons/ninepx.png"), "90px", self)
        self.brushsize.addAction(ninepxAction)
        ninepxAction.triggered.connect(self.ninePixel)

        twelvepxAction = QAction(QIcon("icons/twelvepx.png"), "120px", self)
        self.brushsize.addAction(twelvepxAction)
        twelvepxAction.triggered.connect(self.twelvePixel)

        fifteenpxAction = QAction(QIcon("icons/twelvepx.png"), "150px", self)
        self.brushsize.addAction(fifteenpxAction)
        fifteenpxAction.triggered.connect(self.fifteenPixel)

        eighteenAction = QAction(QIcon("icons/fivepx.png"), "180px", self)
        self.brushsize.addAction(eighteenAction)
        eighteenAction.triggered.connect(self.eighteenPixel)

        twentyoneAction = QAction(QIcon("icons/fivepx.png"), "210px", self)
        self.brushsize.addAction(twentyoneAction)
        twentyoneAction.triggered.connect(self.twentyonePixel)

        twentyfourAction = QAction(QIcon("icons/fivepx.png"), "240px", self)
        self.brushsize.addAction(twentyfourAction)
        twentyfourAction.triggered.connect(self.twentyfourPixel)

        twentysevenAction = QAction(QIcon("icons/fivepx.png"), "270px", self)
        self.brushsize.addAction(twentysevenAction)
        twentysevenAction.triggered.connect(self.twentysevenPixel)

        thAction = QAction(QIcon("icons/fivepx.png"), "300px", self)
        self.brushsize.addAction(thAction)
        thAction.triggered.connect(self.thPixel)

        ththAction = QAction(QIcon("icons/fivepx.png"), "330px", self)
        self.brushsize.addAction(ththAction)
        ththAction.triggered.connect(self.ththPixel)

        thsixAction = QAction(QIcon("icons/fivepx.png"), "360px", self)
        self.brushsize.addAction(thsixAction)
        thsixAction.triggered.connect(self.thsixPixel)

        self.show()


    def twPixel(self):
        self.brushSize=2

    def threePixel(self):
        self.brushSize=3

    def sixPixel(self):
        self.brushSize=6

    def ninePixel(self):
        self.brushSize=9

    def twelvePixel(self):
        self.brushSize=12

    def fifteenPixel(self):
        self.brushSize=15

    def eighteenPixel(self):
        self.brushSize=18

    def twentyonePixel(self):
        self.brushSize=21

    def twentyfourPixel(self):
        self.brushSize=24

    def twentysevenPixel(self):
        self.brushSize=27

    def thPixel(self):
        self.brushSize=30

    def ththPixel(self):
        self.brushSize=33

    def thsixPixel(self):
        self.brushSize=36

    def autocorrect(self):
        try:

            img = cv2.imread(self.filename)
            img = simplest_cb(img, 0.005)
            qimage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap = QtGui.QPixmap(qimage).scaled(self.label.size(),
                                                  QtCore.Qt.KeepAspectRatio)
            self.image = pixmap
            self.label.setPixmap(self.image)
        except:
            pass
    ''''''
    def mousePressEvent(self, event):
        try:
            #
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.lastPoint = event.pos()-QPoint(self.label.x(), self.label.y()) #21,111
                #self.prvs = self.label.pixmap()
                print(self.lastPoint)
                #print(self.label.size())

                if self.count==0:
                    self.prvs1 = self.label.pixmap().copy()
                    self.count = 1
                else:
                    self.prvs2 = self.label.pixmap().copy()
                    self.count =0
                print(self.count)
                self.update()

        except:
            pass

    def mouseMoveEvent(self, event):
        try:

            if event.buttons() and Qt.LeftButton and self.drawing:

                painter = QPainter(self.label.pixmap()) #self.image #self.label.pixmap()
                #painter.setOpacity(3 * 0.01)
                painter.setPen(QPen(Qt.green, self.brushSize, Qt.SolidLine)) #Qt.white

                painter.drawLine(self.lastPoint, event.pos()-QPoint(self.label.x(), self.label.y())) #21,111
                #self.label.setPixmap(self.image)
                self.lastPoint = event.pos()-QPoint(self.label.x(), self.label.y()) #21,111
                #print(self.lastPoint)

                painter.end()

                #self.drawing = False

                self.label.setPixmap(self.label.pixmap())

        except:
            pass


    #def mouseReleaseEvent(self, event):
        #if event.button == Qt.LeftButton:
            #self.drawing = False
        #https://stackoverflow.com/questions/64874382/pyqt5-undo-implementation

    def undo(self):
        try:
            print(self.count)
            if self.count == 1:
                pixmap = self.prvs1
                print("prvs1")
            else:
                pixmap = self.prvs2
                print("prvs2")
        except:
            print("error")
            pass #pixmap=self.label.pixmap()
        self.label.setPixmap(pixmap)
        self.update()


    def clear(self):
        if self.filename:
            pixmap = QtGui.QPixmap(self.filename).scaled(self.label.size(),
                                                         QtCore.Qt.KeepAspectRatio)
            if pixmap.isNull():
                return
            self.image = pixmap
            #print(self.image.size())
            self.prvs1 = pixmap.copy()
            self.prvs2 = pixmap.copy()
            self.label.setPixmap(self.image)
            self.overlap_mask=0


    def loadImage(self):
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        if self.filename:

            pixmap = QtGui.QPixmap(self.filename).scaled(self.label.size(),
                QtCore.Qt.KeepAspectRatio)
            if pixmap.isNull():
                return
            self.image = pixmap
            #print(self.image.size())
            self.label.setPixmap(self.image)
            self.prvs1 = pixmap.copy()
            self.prvs2 = pixmap.copy()
            dirpath = os.path.dirname(self.filename)
            self.fileList = []
            for f in os.listdir(dirpath):
                fpath = os.path.join(dirpath, f)
                if os.path.isfile(fpath) and f.endswith(('.png', '.jpg', '.jpeg')):
                    self.fileList.append(fpath)
            self.fileList.sort()
            self.dirIterator = iter(self.fileList)


    def nextImage(self):
        # ensure that the file list has not been cleared due to missing files
        if self.fileList:
            try:
                self.filename = next(self.dirIterator)

                pixmap = QtGui.QPixmap(self.filename).scaled(self.label.size(),
                    QtCore.Qt.KeepAspectRatio)
                if pixmap.isNull():
                    # the file is not a valid image, remove it from the list
                    # and try to load the next one
                    self.fileList.remove(self.filename)
                    self.nextImage()
                else:
                    self.image = pixmap
                    #print(self.image.size())
                    self.label.setPixmap(self.image)
            except:
                # the iterator has finished, restart it
                self.dirIterator = iter(self.fileList)
                self.nextImage()
        else:
            # no file list found, load an image
            self.loadImage()


    def mask(self):
        try:
            #self.m = mask_thresh(self.filename)
            #print("1")
            #m = m[:, :, 0]
            #print("2")
            img = cv2.imread(self.filename)
            img = simplest_cb(img,0.005)
            #print("3")
            mask_ext = np.expand_dims(self.m, axis=2)
            M = np.concatenate((mask_ext, mask_ext, np.zeros(mask_ext.shape, dtype="uint8")), axis=2)
            #ii = M*255
            #ii = cv2.bitwise_and(img,img,mask = self.m)

            ii = cv2.add(img,M*255) #ii)

            #ii = color.label2rgb(self.m, image=img,bg_label=0)
            #print("4")

            qimage = QtGui.QImage(ii.data, ii.shape[1], ii.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            #print("5")
            pixmap = QtGui.QPixmap(qimage).scaled(self.label.size(),
                                                  QtCore.Qt.KeepAspectRatio)

            #print("6")
            self.image = pixmap
            self.label.setPixmap(self.image)
            self.overlap_mask=1
        except:
            print("Didn't work!")
            pass


    def length_calcul(self):
        Im = self.label.pixmap()
        im = Im.toImage()
        #print(im)


        #print(ar)
        s = im.bits().asstring(im.size().width()*im.size().height()*im.depth() // 8 )
        ar = np.frombuffer(s, dtype=np.uint8).reshape(im.size().height(), im.size().width(), im.depth() // 8)
        #plt.imshow(ar)
        #plt.show()
        hsv = cv2.cvtColor(ar, cv2.COLOR_BGR2HSV)
        #plt.imshow(hsv)
        #plt.show()
        # https://stackoverflow.com/questions/31460267/python-opencv-color-tracking
        lower_white = np.array([30, 100, 100], dtype=np.uint8)
        upper_white = np.array([70, 255, 255], dtype=np.uint8)

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        print("thresh")
        m = mask > 0
        m = morphology.remove_small_objects(m, min_size=178, connectivity=1).astype("uint8")
        print(m.shape)
        #print(self.m.shape)
        if self.overlap_mask == 1:
            try:
                mask = m + cv2.resize(self.m,(m.shape[1],m.shape[0]))
                print("summing masks")
            except:
                mask = m
        else:
            mask = m

        #plt.imshow(mask)
        #plt.show()
        self.M=mask
        #print(self.M.shape)
        skeleton = morphology.medial_axis(mask) #self.m
        sk = round(skeleton.sum()*2 / 103, 1)
        #sk=0
        return sk

    def update_length(self):
        self.length.display(self.length_calcul())

    def save_mask(self):
        try:
            self.update_length()
            name = self.filename[:-4] + "_mask_" + "mm_.jpg"
        except:
            name = ""

        name, filter_ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save file', name, 'MR files (*.jpg)')
        try:
            mask = np.stack((self.M,self.M,self.M),axis=2)
            mask = mask > 0.5
            mask = mask * 255
            mask = mask.astype("uint8")
            cv2.imwrite(name,mask)
            print("done")
        except:
            print("failed")
            pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    imageLoader = ImageLoader()
    imageLoader.show()
    sys.exit(app.exec_())
