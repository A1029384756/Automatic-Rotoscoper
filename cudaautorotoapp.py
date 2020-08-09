from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QProgressBar
import tkinter as tk
import os
import shutil
from tkinter import filedialog
from tkinter import simpledialog as sd
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import PIL
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models

fcn = None

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Ui_MainWindow(object):   
    People = 255
    Vehicles = 0
    Animals = 0
    Other = 0

    def resetBar(self):
        self.step = 0
        self.progressBar.setValue(0)
        return()

    def running(self):
        while self.progress < 100:
            self.progressBar.setValue(self.progress)
            return()
        else:
            self.resetBar()
            return()

    def Person(self, state):
        if QtCore.Qt.Checked == state:
            self.People = 255
            print("On")
        elif QtCore.Qt.Checked != state:
            self.People = 0
            print("Off")

    def Vehicle(self, state):
        if QtCore.Qt.Checked == state:
            self.Vehicles = 255
            print("On")
        elif QtCore.Qt.Checked != state:
            self.Vehicles = 0
            print("Off")

    def Animal(self, state):
        if QtCore.Qt.Checked == state:
            self.Animals = 255
            print("On")
        elif QtCore.Qt.Checked != state:
            self.Animals = 0
            print("Off")

    def thing(self, state):
        if QtCore.Qt.Checked == state:
            self.Other = 255
            print("On")
        elif QtCore.Qt.Checked != state:
            self.Other = 0
            print("Off")

    def FilePath(self):
        self.pushButton_3.setEnabled(True)
        root = tk.Tk()
        root.withdraw()
        self.directoryName = filedialog.askdirectory(parent=root, initialdir="/", title = 'Please select a directory')
        if self.directoryName != "":
            self.listOfFiles = [f for f in listdir(self.directoryName) if isfile(join(self.directoryName, f))]
            if os.path.isdir(self.directoryName + "/Output") == True:
                shutil.rmtree(self.directoryName + "/Output")
                os.mkdir(self.directoryName + "/Output")
                self.outputDirectory = self.directoryName + "/Output"
            else:
                os.mkdir(self.directoryName + "/Output")
                self.outputDirectory = self.directoryName + "/Output"

    def ShowPopup(self):
        self.buttonEnableRes = 0
        msg = QMessageBox()
        msg.setWindowTitle("Resolution Alert")
        msg.setText("The resolution you provide must be greater than zero")
        msg.setStandardButtons(QMessageBox.Ok)

        x = msg.exec_()

    def ShowInvalidPopup(self):
        self.buttonEnableRes = 0
        msg = QMessageBox()
        msg.setWindowTitle("Resolution Alert")
        msg.setText("Invalid resolution input")
        msg.setStandardButtons(QMessageBox.Ok)

        x = msg.exec_()        

    def getRotoModel(self):
        global fcn
        fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
        if torch.cuda.is_available():
            fcn.cuda()

    def decode_segmap(self, image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (self.Vehicles, self.Vehicles, self.Vehicles), (self.Vehicles, self.Vehicles, self.Vehicles), (self.Animals, self.Animals, self.Animals), (self.Vehicles, self.Vehicles, self.Vehicles), (self.Other, self.Other, self.Other),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (self.Vehicles, self.Vehicles, self.Vehicles), (self.Vehicles, self.Vehicles, self.Vehicles), (self.Animals, self.Animals, self.Animals), (self.Other, self.Other, self.Other), (self.Animals, self.Animals, self.Animals),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (self.Other, self.Other, self.Other), (self.Animals, self.Animals, self.Animals), (self.Animals, self.Animals, self.Animals), (self.Vehicles, self.Vehicles, self.Vehicles), (self.People, self.People, self.People),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (self.Other, self.Other, self.Other), (self.Animals, self.Animals, self.Animals), (self.Other, self.Other, self.Other), (self.Vehicles, self.Vehicles, self.Vehicles), (self.Other, self.Other, self.Other)])

        r =  np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def createMatte(self, filename, matteName, size):
        img = Image.open(filename)
        Width, Height = img.size 
        trf = T.Compose([T.Resize(size),
                        T.ToTensor(), 
                        T.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)
        if torch.cuda.is_available():
            inp = inp.cuda()
        with torch.no_grad():
            if (fcn == None): self.getRotoModel()
            out = fcn(inp)['out']
            om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()  
            rgb = self.decode_segmap(om)
            im = Image.fromarray(rgb)
            im = im.resize((Width, Height))
            im.save(matteName)

    def ResolutionChecker(self):
        self.matteHeight = self.lineEdit.text()
        if not self.matteHeight:
            print("Showpop")
            self.ShowInvalidPopup()
            return()
        if int(self.matteHeight) < 1:
            print("Showpop")
            self.ShowPopup()
        else:
            self.matteHeight = int(self.lineEdit.text())
            self.pushButton_2.setEnabled(True)
        print(self.matteHeight)

    def Rotoscope(self):
        i = 1
        print("pressed")
        for currentFile in self.listOfFiles:
            length = len(self.listOfFiles)
            sourceFile = self.directoryName + "/" + currentFile
            mainNameEnd = currentFile.find('.')
            nameForMatte = currentFile[:mainNameEnd] + "_matte" + currentFile[mainNameEnd:]
            fullPathMatteName = self.outputDirectory + "/" + nameForMatte
            self.createMatte(sourceFile, fullPathMatteName, self.matteHeight)
            print("Just created: " + nameForMatte)
            i+=1
            self.progress = 100.0*i/length
            self.running()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(200, 350)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 30, 121, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 150, 141, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(30, 220, 70, 17))
        self.checkBox.setChecked(True)
        self.checkBox.setTristate(False)
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(30, 240, 70, 17))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(30, 260, 70, 17))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(30, 280, 111, 17))
        self.checkBox_4.setObjectName("checkBox_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 200, 121, 16))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(30, 110, 61, 16))
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 80, 91, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(100, 110, 71, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(25, 310, 180, 15)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #connect buttons and boxes to functions
        self.pushButton.clicked.connect(self.FilePath)
        self.lineEdit.returnPressed.connect(self.ResolutionChecker)
        self.pushButton_3.clicked.connect(self.ResolutionChecker)
        self.pushButton_2.clicked.connect(self.Rotoscope)
        self.checkBox.stateChanged.connect(self.Person)
        self.checkBox_2.stateChanged.connect(self.Vehicle)
        self.checkBox_3.stateChanged.connect(self.Animal)
        self.checkBox_4.stateChanged.connect(self.thing)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Auto Rotoscoper"))
        self.pushButton.setText(_translate("MainWindow", "Select Input Images"))
        self.pushButton_2.setText(_translate("MainWindow", "Start Rotoscoping"))
        self.checkBox.setText(_translate("MainWindow", "People"))
        self.checkBox_2.setText(_translate("MainWindow", "Vehicles"))
        self.checkBox_3.setText(_translate("MainWindow", "Animals"))
        self.checkBox_4.setText(_translate("MainWindow", "Household Objects"))
        self.label.setText(_translate("MainWindow", "Create Matte For:"))
        self.label_2.setText(_translate("MainWindow", "Set Matte Height:"))
        self.pushButton_3.setText(_translate("MainWindow", "Set"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())