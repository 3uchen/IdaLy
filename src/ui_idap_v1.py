#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：3uchen time:2022/12/5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

font = QtGui.QFont()
font.setFamily("Microsoft YaHei UI")
font.setPointSize(9)
font.setWeight(440)


class Ui_Gen_Visual:

    def setupui(self, Gen_Visual):
        Gen_Visual.resize(600, 800)

        self.witdget_head = QtWidgets.QWidget(Gen_Visual)
        self.witdget_head.setMaximumSize(QtCore.QSize(16775, 30))
        self.witdget_head.setMinimumSize(QtCore.QSize(600, 30))
        self.witdget_head.setStyleSheet('''QWidget#widget{
                   border-image:url(:/head.png);
                   }''')
        self.logo = QtWidgets.QLabel(self.witdget_head)
        self.logo.move(3, 3)
        self.logo.setMaximumSize(QtCore.QSize(24, 24))
        self.logo.setMinimumSize(QtCore.QSize(24, 24))
        self.label_train = QtWidgets.QLabel(self.witdget_head)
        self.label_train.move(32, 3)
        self.pix = QPixmap(":/logo.png")
        self.logo.setPixmap(self.pix)
        self.logo.setScaledContents(True)
        self.label_train.setText('Model Training')
        self.label_train.setStyleSheet('''
                                    font-family:"Microsoft YaHei UI";
                                   font-size:20px;
                                   font-weight:bold;
                                   color:#FFFFFF;
                                   line-height:39px
                                   ''')
        self.widget_main = QtWidgets.QWidget(Gen_Visual)
        self.widget_main.setGeometry(QtCore.QRect(10, 40, 600, 1000))
        self.pb = QtWidgets.QProgressBar(self.widget_main)
        self.pb.move(10, 10)
        self.pb.resize(300, 30)
        self.pb.setStyleSheet(
            "QProgressBar { border: 2px solid grey; border-radius: 5px; background-color: #FFFFFF; text-align: center;}QProgressBar::chunk {background:QLinearGradient(x1:0,y1:0,x2:2,y2:0,stop:0 #666699,stop:1  #DB7093); }"
        )
        self.pb.setMaximum(100)
        self.pb.setMinimum(0)
        self.pushButton_stop = QtWidgets.QPushButton(self.widget_main)
        self.pushButton_stop.move(360, 10)
        self.pushButton_stop.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_stop.setMaximumSize(QtCore.QSize(100, 40))
        self.pushButton_stop.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;font-weight:500;font-size:16px}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_stop.setText("Stop")
        self.pushButton_stop.setFont(font)
        self.PCADisplay = QtWidgets.QGroupBox(self.widget_main)
        self.PCADisplay.move(10, 300)
        self.PCADisplay.setStyleSheet('''border:1px solid #FFFFFF ''')
        self.fig_visual = plt.Figure(figsize=(5, 3))
        self.visual_canva = FigureCanvas(self.fig_visual)
        self.axes_visual = self.visual_canva.figure.subplots()
        self.fig_visual.subplots_adjust(top=0.985,
                                        bottom=0.085,
                                        left=0.090,
                                        right=0.975,
                                        hspace=0.2,
                                        wspace=0.2)
        self.layout_visual = QtWidgets.QVBoxLayout()
        self.layout_visual.addWidget(self.visual_canva)
        self.PCADisplay.setLayout(self.layout_visual)


class Ui_Form(object):

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1360, 800)
        Form.setStyleSheet("#Form{border-image:url(head.png);}")
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(14)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setMaximumSize(QtCore.QSize(167755555, 80))
        self.widget.setMinimumSize(QtCore.QSize(1280, 80))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(22, 20,
                                           QtWidgets.QSizePolicy.Maximum,
                                           QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(64, 64))
        self.label.setMinimumSize(QtCore.QSize(64, 64))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem1 = QtWidgets.QSpacerItem(24, 20,
                                            QtWidgets.QSizePolicy.Maximum,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        spacerItem21 = QtWidgets.QSpacerItem(100, 20,
                                             QtWidgets.QSizePolicy.Expanding,
                                             QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem21)
        self.label_open = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label_open.sizePolicy().hasHeightForWidth())
        self.label_open.setSizePolicy(sizePolicy)
        self.label_open.setObjectName("label_open")
        self.label_open.setMaximumSize(QtCore.QSize(32, 32))
        self.label_open.setMinimumSize(QtCore.QSize(32, 32))
        self.label_open.setToolTip("Open File")
        self.label_open.setToolTipDuration(3000)
        self.horizontalLayout.addWidget(self.label_open)
        spacerItem31 = QtWidgets.QSpacerItem(24, 20,
                                             QtWidgets.QSizePolicy.Maximum,
                                             QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem31)
        self.label_save = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label_save.sizePolicy().hasHeightForWidth())
        self.label_save.setSizePolicy(sizePolicy)
        self.label_save.setObjectName("label_save")
        self.label_save.setMaximumSize(QtCore.QSize(32, 32))
        self.label_save.setMinimumSize(QtCore.QSize(32, 32))
        self.label_save.setToolTip("Save All")
        self.label_save.setToolTipDuration(3000)
        self.horizontalLayout.addWidget(self.label_save)
        spacerItem41 = QtWidgets.QSpacerItem(24, 20,
                                             QtWidgets.QSizePolicy.Maximum,
                                             QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem41)
        self.label_help = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label_help.sizePolicy().hasHeightForWidth())
        self.label_help.setSizePolicy(sizePolicy)
        self.label_help.setObjectName("label")
        self.label_help.setToolTip("Help")
        self.label_help.setToolTipDuration(3000)
        self.label_help.setMaximumSize(QtCore.QSize(32, 32))
        self.label_help.setMinimumSize(QtCore.QSize(32, 32))
        self.horizontalLayout.addWidget(self.label_help)
        spacerItem1 = QtWidgets.QSpacerItem(280, 20,
                                            QtWidgets.QSizePolicy.Maximum,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(20)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 0))
        self.pushButton.setMaximumSize(QtCore.QSize(20, 20))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        spacerItem4 = QtWidgets.QSpacerItem(8, 10,
                                            QtWidgets.QSizePolicy.Maximum,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.pushButton_3 = QtWidgets.QPushButton(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(20)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setMaximumSize(QtCore.QSize(20, 20))
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        spacerItem5 = QtWidgets.QSpacerItem(20, 10,
                                            QtWidgets.QSizePolicy.Maximum,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.verticalLayout.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(Form)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_2.setContentsMargins(14, 0, 14, 14)
        self.horizontalLayout_2.setSpacing(14)
        self.horizontalLayout_2.setObjectName("horizontalLayout")

        self.widget_3 = QtWidgets.QWidget(Form)
        self.widget_3.setObjectName("widget_3")
        self.widget_3.setMaximumSize(QtCore.QSize(228, 11111111))
        self.widget_3.setMinimumSize(QtCore.QSize(228, 692))
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_1.setContentsMargins(14, 15, 14, 12)
        self.verticalLayout_1.setSpacing(0)
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.widget_method = QtWidgets.QWidget(Form)
        self.widget_method.setObjectName("widget_method")
        self.widget_method.setMaximumSize(QtCore.QSize(165, 15))
        self.widget_method.setMinimumSize(QtCore.QSize(165, 15))
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_method)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(9)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget_method)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMaximumSize(QtCore.QSize(14, 14))
        self.label_3.setMinimumSize(QtCore.QSize(14, 14))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.widget_method)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.verticalLayout_1.addWidget(self.widget_method)
        spacerItem7 = QtWidgets.QSpacerItem(1, 15,
                                            QtWidgets.QSizePolicy.Maximum,
                                            QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_1.addItem(spacerItem7)
        self.widget_toolbox = QtWidgets.QWidget(Form)
        self.widget_toolbox.setObjectName("widget_toolbox")
        self.widget_toolbox.setMaximumSize(QtCore.QSize(400, 500))
        self.widget_toolbox.setMinimumSize(QtCore.QSize(600, 500))
        self.toolBox = QtWidgets.QToolBox(self.widget_toolbox)
        self.toolBox.setMinimumSize(QtCore.QSize(200, 400))
        self.toolBox.setMaximumSize(QtCore.QSize(200, 400))
        self.toolBox.setLineWidth(20)
        self.toolBox.setObjectName("toolBox")
        self.page_T1 = QtWidgets.QWidget(Form)
        self.page_T1.setMinimumSize(QtCore.QSize(180, 230))
        self.page_T1.setMaximumSize(QtCore.QSize(180, 230))
        self.page_T1.setStyleSheet(
            " background-color:#ffffff;\n"
            "                   border:1px solid #CCCCCC;\n"
            "                   border-radius:6px")
        self.page_T1.setObjectName("page_T1")
        self.layout_T1 = QtWidgets.QVBoxLayout()
        self.layout_T1.setContentsMargins(18, 0, 15, 0)
        self.layout_T1.setObjectName("layout_T1")
        self.scroll1 = QtWidgets.QScrollArea()
        self.page_T1.setLayout(self.layout_T1)
        self.pushButton_GNI = QtWidgets.QPushButton()
        self.pushButton_GNI.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_GNI.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_GNI.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_GNI.setText("1. Gaussian Noise")
        self.pushButton_GNI.setFont(font)
        self.pushButton_GNI.setObjectName("pushButton_GNI")
        self.layout_T1.addWidget(self.pushButton_GNI)
        self.pushButton_MSI = QtWidgets.QPushButton()
        self.pushButton_MSI.setObjectName("pushButton_MSI")
        self.pushButton_MSI.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_MSI.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_MSI.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_MSI.setText("2. Masking Noise")
        self.pushButton_MSI.setFont(font)
        self.layout_T1.addWidget(self.pushButton_MSI)
        self.pushButton_SNI = QtWidgets.QPushButton()
        self.pushButton_SNI.setObjectName("pushButton_SNI")
        self.pushButton_SNI.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_SNI.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_SNI.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_SNI.setText("3. Speckle Noise")
        self.pushButton_SNI.setFont(font)
        self.layout_T1.addWidget(self.pushButton_SNI)
        self.pushButton_PNI = QtWidgets.QPushButton()
        self.pushButton_PNI.setObjectName("pushButton_PNI")
        self.pushButton_PNI.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_PNI.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_PNI.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_PNI.setText("4. Poisson Noise")
        self.pushButton_PNI.setFont(font)
        self.layout_T1.addWidget(self.pushButton_PNI)
        self.pushButton_COUTOUT = QtWidgets.QPushButton()
        self.pushButton_COUTOUT.setObjectName("pushButton_PNI")
        self.pushButton_COUTOUT.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_COUTOUT.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_COUTOUT.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_COUTOUT.setText("5. Cutout")
        self.pushButton_COUTOUT.setFont(font)
        self.layout_T1.addWidget(self.pushButton_COUTOUT)
        self.scroll1.setWidget(self.page_T1)
        self.toolBox.addItem(self.scroll1, "")

        self.page_T2 = QtWidgets.QWidget()
        self.page_T2.setObjectName("page_T2")
        self.page_T2.setMinimumSize(QtCore.QSize(180, 320))
        self.page_T2.setMaximumSize(QtCore.QSize(180, 320))
        self.page_T2.setStyleSheet(
            " background-color:#ffffff;\n"
            "                   border:1px solid #CCCCCC;\n"
            "                   border-radius:6px")
        self.layout_T2 = QtWidgets.QVBoxLayout()
        self.layout_T2.setContentsMargins(15, 0, 15, 0)
        self.layout_T2.setObjectName("layout_T2")
        self.scroll2 = QtWidgets.QScrollArea()
        self.page_T2.setLayout(self.layout_T2)
        self.pushButton_SMOTE = QtWidgets.QPushButton()
        self.pushButton_SMOTE.setObjectName("pushButton_SMOTE")
        self.pushButton_SMOTE.setMinimumSize(QtCore.QSize(150, 35))
        self.pushButton_SMOTE.setMaximumSize(QtCore.QSize(150, 35))
        self.pushButton_SMOTE.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_SMOTE.setText("1. SMOTE")
        self.pushButton_SMOTE.setFont(font)
        self.layout_T2.addWidget(self.pushButton_SMOTE)
        self.pushButton_KMEANS_SMOTE = QtWidgets.QPushButton()
        self.pushButton_KMEANS_SMOTE.setObjectName("pushButton_KMEANS_SMOTE")
        self.pushButton_KMEANS_SMOTE.setMinimumSize(QtCore.QSize(150, 35))
        self.pushButton_KMEANS_SMOTE.setMaximumSize(QtCore.QSize(150, 35))
        self.pushButton_KMEANS_SMOTE.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_KMEANS_SMOTE.setText("2. KMEANS_SMOTE")
        self.pushButton_KMEANS_SMOTE.setFont(font)
        self.layout_T2.addWidget(self.pushButton_KMEANS_SMOTE)
        self.pushButton_LLE = QtWidgets.QPushButton()
        self.pushButton_LLE.setObjectName("pushButton_LLE")
        self.pushButton_LLE.setMinimumSize(QtCore.QSize(150, 35))
        self.pushButton_LLE.setMaximumSize(QtCore.QSize(150, 35))
        self.pushButton_LLE.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_LLE.setText("3. LLE")
        self.pushButton_LLE.setFont(font)
        self.layout_T2.addWidget(self.pushButton_LLE)
        self.pushButton_MIXUP = QtWidgets.QPushButton()
        self.pushButton_MIXUP.setObjectName("pushButton_MIXUP")
        self.pushButton_MIXUP.setMinimumSize(QtCore.QSize(150, 35))
        self.pushButton_MIXUP.setMaximumSize(QtCore.QSize(150, 35))
        self.pushButton_MIXUP.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_MIXUP.setText("4. MIXUP")
        self.pushButton_MIXUP.setFont(font)
        self.layout_T2.addWidget(self.pushButton_MIXUP)
        self.pushButton_MTD = QtWidgets.QPushButton()
        self.pushButton_MTD.setObjectName("pushButton_MTD")
        self.pushButton_MTD.setMinimumSize(QtCore.QSize(150, 35))
        self.pushButton_MTD.setMaximumSize(QtCore.QSize(150, 35))
        self.pushButton_MTD.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_MTD.setText("5. MTD")
        self.pushButton_MTD.setFont(font)
        self.layout_T2.addWidget(self.pushButton_MTD)
        self.pushButton_KNNMTD = QtWidgets.QPushButton()
        self.pushButton_KNNMTD.setObjectName("pushButton_KNNMTD")
        self.pushButton_KNNMTD.setMinimumSize(QtCore.QSize(150, 35))
        self.pushButton_KNNMTD.setMaximumSize(QtCore.QSize(150, 35))
        self.pushButton_KNNMTD.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_KNNMTD.setText("6. KNNMTD")
        self.pushButton_KNNMTD.setFont(font)
        self.layout_T2.addWidget(self.pushButton_KNNMTD)
        self.scroll2.setWidget(self.page_T2)
        self.toolBox.addItem(self.scroll2, "")

        self.page_T3 = QtWidgets.QWidget()
        self.page_T3.setObjectName("page_T3")
        self.page_T3.setMinimumSize(QtCore.QSize(180, 440))
        self.page_T3.setMaximumSize(QtCore.QSize(180, 440))
        self.page_T3.setStyleSheet(
            " background-color:#ffffff;\n"
            "                   border:1px solid #CCCCCC;\n"
            "                   border-radius:6px")
        self.layout_T3 = QtWidgets.QVBoxLayout()
        self.layout_T3.setContentsMargins(18, 0, 15, 0)
        self.layout_T3.setObjectName("layout_T3")
        self.scroll3 = QtWidgets.QScrollArea()
        self.page_T3.setLayout(self.layout_T3)
        self.pushButton_GMM = QtWidgets.QPushButton()
        self.pushButton_GMM.setObjectName("pushButton_GMM")
        self.pushButton_GMM.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_GMM.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_GMM.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_GMM.setText("1. GMM")
        self.pushButton_GMM.setFont(font)
        self.layout_T3.addWidget(self.pushButton_GMM)
        self.pushButton_GAN = QtWidgets.QPushButton()
        self.pushButton_GAN.setObjectName("pushButton_GAN")
        self.pushButton_GAN.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_GAN.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_GAN.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_GAN.setText("2. GAN")
        self.pushButton_GAN.setFont(font)
        self.layout_T3.addWidget(self.pushButton_GAN)
        self.pushButton_WGAN_GP = QtWidgets.QPushButton()
        self.pushButton_WGAN_GP.setObjectName("pushButton_WGAN_GP")
        self.pushButton_WGAN_GP.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_WGAN_GP.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_WGAN_GP.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_WGAN_GP.setText("3. WGAN-GP")
        self.pushButton_WGAN_GP.setFont(font)
        self.layout_T3.addWidget(self.pushButton_WGAN_GP)
        self.pushButton_LSGAN = QtWidgets.QPushButton()
        self.pushButton_LSGAN.setObjectName("pushButton_LSGAN")
        self.pushButton_LSGAN.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_LSGAN.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_LSGAN.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_LSGAN.setText("4. LSGAN")
        self.pushButton_LSGAN.setFont(font)
        self.layout_T3.addWidget(self.pushButton_LSGAN)
        self.pushButton_VAE = QtWidgets.QPushButton()
        self.pushButton_VAE.setObjectName("pushButton_VAE")
        self.pushButton_VAE.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_VAE.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_VAE.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_VAE.setText("5. VAE")
        self.pushButton_VAE.setFont(font)
        self.layout_T3.addWidget(self.pushButton_VAE)
        self.pushButton_VAEGAN = QtWidgets.QPushButton()
        self.pushButton_VAEGAN.setObjectName("pushButton_VAEGAN")
        self.pushButton_VAEGAN.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_VAEGAN.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_VAEGAN.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_VAEGAN.setText("6. VAEGAN")
        self.pushButton_VAEGAN.setFont(font)
        self.layout_T3.addWidget(self.pushButton_VAEGAN)
        self.pushButton_MAF = QtWidgets.QPushButton()
        self.pushButton_MAF.setObjectName("pushButton_MAF")
        self.pushButton_MAF.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_MAF.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_MAF.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_MAF.setText("7. MAF")
        self.pushButton_MAF.setFont(font)
        self.layout_T3.addWidget(self.pushButton_MAF)
        self.pushButton_REALNVP = QtWidgets.QPushButton()
        self.pushButton_REALNVP.setObjectName("pushButton_REALNVP")
        self.pushButton_REALNVP.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_REALNVP.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_REALNVP.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_REALNVP.setText("8. REALNVP")
        self.pushButton_REALNVP.setFont(font)
        self.layout_T3.addWidget(self.pushButton_REALNVP)
        self.pushButton_GLOW = QtWidgets.QPushButton()
        self.pushButton_GLOW.setObjectName("pushButton_GLOW")
        self.pushButton_GLOW.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_GLOW.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_GLOW.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_GLOW.setText("9. GLOW")
        self.pushButton_GLOW.setFont(font)
        self.layout_T3.addWidget(self.pushButton_GLOW)
        self.pushButton_DDPM = QtWidgets.QPushButton()
        self.pushButton_DDPM.setObjectName("pushButton_DDPM")
        self.pushButton_DDPM.setMinimumSize(QtCore.QSize(140, 35))
        self.pushButton_DDPM.setMaximumSize(QtCore.QSize(140, 35))
        self.pushButton_DDPM.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_DDPM.setText("10. DDPM")
        self.pushButton_DDPM.setFont(font)
        self.layout_T3.addWidget(self.pushButton_DDPM)
        self.scroll3.setWidget(self.page_T3)
        self.toolBox.addItem(self.scroll3, "")

        self.page_T7 = QtWidgets.QWidget()
        self.page_T7.setGeometry(QtCore.QRect(50, 50, 141, 171))
        self.page_T7.setObjectName("page_T7")
        self.page_T7.setStyleSheet(" background-color:#ffffff;\n")
        self.toolBox.addItem(self.page_T7, "")
        self.verticalLayout_1.addWidget(self.widget_toolbox)

        spacerItem8 = QtWidgets.QSpacerItem(10, 10,
                                            QtWidgets.QSizePolicy.Fixed,
                                            QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_1.addItem(spacerItem8)
        self.widget_description = QtWidgets.QWidget(Form)
        self.widget_description.setObjectName("widget_description")
        self.widget_description.setMaximumSize(QtCore.QSize(140, 20))
        self.widget_description.setMinimumSize(QtCore.QSize(140, 20))
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(
            self.widget_description)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(9)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_5 = QtWidgets.QLabel(self.widget_description)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setMaximumSize(QtCore.QSize(14, 14))
        self.label_5.setMinimumSize(QtCore.QSize(14, 14))
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.widget_description)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.verticalLayout_1.addWidget(self.widget_description)
        spacerItem9 = QtWidgets.QSpacerItem(10, 10,
                                            QtWidgets.QSizePolicy.Fixed,
                                            QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_1.addItem(spacerItem9)
        self.plainTextEdit = QtWidgets.QTextEdit(self.widget_3)
        self.plainTextEdit.setMinimumSize(200, 190)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.plainTextEdit.setStyleSheet(
            " background-color:#ffffff;\n"
            "                   border:1px solid #CCCCCC;\n"
            "                   border-radius:6px")
        option = QtGui.QTextOption()
        option.setAlignment(QtCore.Qt.AlignJustify)
        self.plainTextEdit.document().setDefaultTextOption(option)
        self.plainTextEdit.setFont(QtGui.QFont("Times New Roman", 11))
        self.verticalLayout_1.addWidget(self.plainTextEdit)

        self.horizontalLayout_2.addWidget(self.widget_3)

        self.widget_4 = QtWidgets.QWidget(Form)
        self.widget_4.setObjectName("widget_4")
        self.widget_4.setMaximumSize(QtCore.QSize(167755555, 11111111))
        self.widget_4.setMinimumSize(QtCore.QSize(534, 692))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(14)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_41 = QtWidgets.QWidget(Form)
        self.widget_41.setObjectName("widget_41")
        self.widget_41.setMaximumSize(QtCore.QSize(167755555, 175))
        self.widget_41.setMinimumSize(QtCore.QSize(534, 175))
        self.widget_para = QtWidgets.QWidget(self.widget_41)
        self.widget_para.setObjectName("widget_para")
        self.widget_para.setGeometry(QtCore.QRect(15, 10, 194, 30))
        self.horizontalLayout_41 = QtWidgets.QHBoxLayout(self.widget_para)
        self.horizontalLayout_41.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_41.setSpacing(9)
        self.horizontalLayout_41.setObjectName("horizontalLayout_41")
        self.label_7 = QtWidgets.QLabel(self.widget_para)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setMaximumSize(QtCore.QSize(14, 14))
        self.label_7.setMinimumSize(QtCore.QSize(14, 14))
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_41.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.widget_para)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_41.addWidget(self.label_8)
        self.stackedWidget = QtWidgets.QStackedWidget(self.widget_41)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 25, 530, 150))
        self.page_0 = QtWidgets.QWidget()
        self.stackedWidget.addWidget(self.page_0)

        self.page_GNI = QtWidgets.QWidget()
        self.HLayout_GNI = QtWidgets.QHBoxLayout(self.page_GNI)
        self.HLayout_GNI.setSpacing(12)
        self.verticalLayout_GNI1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_GNI1.setSpacing(0)
        self.verticalLayout_GNI1.setObjectName("verticalLayout_GNI1")
        self.mean_para = QtWidgets.QLabel()
        self.mean_para.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                         line-height:26px ''')
        self.mean_para.setText("Mean")
        self.verticalLayout_GNI1.addWidget(self.mean_para)
        self.mean_value = QtWidgets.QDoubleSpinBox(self.page_GNI)
        self.mean_value.setAlignment(QtCore.Qt.AlignCenter)
        self.mean_value.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.mean_value.setMinimumSize(QtCore.QSize(140, 35))
        self.mean_value.setMaximumSize(QtCore.QSize(140, 35))
        self.verticalLayout_GNI1.addWidget(self.mean_value)
        spacerItem_11 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_GNI1.addItem(spacerItem_11)
        self.HLayout_GNI.addLayout(self.verticalLayout_GNI1)
        self.verticalLayout_GNI2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_GNI2.setSpacing(0)
        self.var_para = QtWidgets.QLabel()
        self.var_para.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                                 line-height:26px '''
        )
        self.var_para.setText("Variance")
        self.verticalLayout_GNI2.addWidget(self.var_para)
        self.var_value = QtWidgets.QDoubleSpinBox(self.page_GNI)
        self.var_value.setAlignment(QtCore.Qt.AlignCenter)
        self.var_value.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.var_value.setMinimumSize(QtCore.QSize(140, 35))
        self.var_value.setMaximumSize(QtCore.QSize(140, 35))
        self.verticalLayout_GNI2.addWidget(self.var_value)
        spacerItem_12 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_GNI2.addItem(spacerItem_12)
        self.HLayout_GNI.addLayout(self.verticalLayout_GNI2)
        self.stackedWidget.addWidget(self.page_GNI)

        self.page_MSI = QtWidgets.QWidget()
        self.page_MSI.setGeometry(QtCore.QRect(0, 10, 530, 10))
        self.HLayout_MSI = QtWidgets.QHBoxLayout(self.page_MSI)
        self.HLayout_MSI.setSpacing(12)
        self.verticalLayout_MSI = QtWidgets.QVBoxLayout()
        self.verticalLayout_MSI.setSpacing(0)
        self.verticalLayout_MSI.setObjectName("verticalLayout_MSI")
        self.mp_para_msi = QtWidgets.QLabel()
        self.mp_para_msi.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                                 line-height:26px '''
        )
        self.mp_para_msi.setText("Making Probability")
        self.verticalLayout_MSI.addWidget(self.mp_para_msi)
        self.mp_value_msi = QtWidgets.QDoubleSpinBox(self.page_MSI)
        self.mp_value_msi.setMinimumSize(QtCore.QSize(140, 35))
        self.mp_value_msi.setMaximumSize(QtCore.QSize(140, 35))
        self.mp_value_msi.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.mp_value_msi.setAlignment(QtCore.Qt.AlignCenter)
        self.mp_value_msi.setMaximum(1000000)
        self.verticalLayout_MSI.addWidget(self.mp_value_msi)
        spacerItem_13 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_MSI.addItem(spacerItem_13)
        self.HLayout_MSI.addLayout(self.verticalLayout_MSI)
        self.stackedWidget.addWidget(self.page_MSI)
        self.page_Cutout = QtWidgets.QWidget()
        self.page_Cutout.setGeometry(QtCore.QRect(0, 10, 530, 10))
        self.HLayout_Cutout = QtWidgets.QHBoxLayout(self.page_Cutout)
        self.HLayout_Cutout.setSpacing(12)
        self.verticalLayout_Cutout = QtWidgets.QVBoxLayout()
        self.verticalLayout_Cutout.setSpacing(0)
        self.verticalLayout_Cutout.setObjectName("verticalLayout_Cutout")
        self.mp_para_cutout = QtWidgets.QLabel()
        self.mp_para_cutout.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                                 line-height:26px '''
        )
        self.mp_para_cutout.setText("Cuting Ratio")
        self.verticalLayout_Cutout.addWidget(self.mp_para_cutout)
        self.mp_value_cutout = QtWidgets.QDoubleSpinBox(self.page_Cutout)
        self.mp_value_cutout.setMinimumSize(QtCore.QSize(140, 35))
        self.mp_value_cutout.setMaximumSize(QtCore.QSize(140, 35))
        self.mp_value_cutout.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.mp_value_cutout.setAlignment(QtCore.Qt.AlignCenter)
        self.mp_value_cutout.setMaximum(1000000)
        self.verticalLayout_Cutout.addWidget(self.mp_value_cutout)
        spacerItem_14 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_Cutout.addItem(spacerItem_14)
        self.HLayout_Cutout.addLayout(self.verticalLayout_Cutout)
        self.stackedWidget.addWidget(self.page_Cutout)

        self.page_SMOTE = QtWidgets.QWidget()
        self.page_SMOTE.setGeometry(QtCore.QRect(0, 10, 530, 10))
        self.HLayout_SMOTE = QtWidgets.QHBoxLayout(self.page_SMOTE)
        self.HLayout_SMOTE.setSpacing(12)
        self.verticalLayout_SMOTE = QtWidgets.QVBoxLayout()
        self.verticalLayout_SMOTE.setSpacing(0)
        self.verticalLayout_SMOTE.setObjectName("verticalLayout_SMOTE")
        self.kn_para_smote = QtWidgets.QLabel()
        self.kn_para_smote.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                                 line-height:26px '''
        )
        self.kn_para_smote.setText("Number of Neighbours")
        self.verticalLayout_SMOTE.addWidget(self.kn_para_smote)
        self.kn_value_smote = QtWidgets.QSpinBox(self.page_SMOTE)
        self.kn_value_smote.setMinimumSize(QtCore.QSize(140, 35))
        self.kn_value_smote.setMaximumSize(QtCore.QSize(140, 35))
        self.kn_value_smote.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.kn_value_smote.setAlignment(QtCore.Qt.AlignCenter)
        self.kn_value_smote.setMaximum(1000000)
        self.verticalLayout_SMOTE.addWidget(self.kn_value_smote)
        spacerItem_13 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_SMOTE.addItem(spacerItem_13)
        self.HLayout_SMOTE.addLayout(self.verticalLayout_SMOTE)
        self.stackedWidget.addWidget(self.page_SMOTE)

        self.page_KMEANS_SMOTE = QtWidgets.QWidget()
        self.page_KMEANS_SMOTE.setGeometry(QtCore.QRect(0, 10, 530, 10))
        self.HLayout_KMEANS_SMOTE = QtWidgets.QHBoxLayout(self.page_KMEANS_SMOTE)
        self.HLayout_KMEANS_SMOTE.setSpacing(12)
        self.verticalLayout_KMEANS_SMOTE = QtWidgets.QVBoxLayout()
        self.verticalLayout_KMEANS_SMOTE.setSpacing(0)
        self.verticalLayout_KMEANS_SMOTE.setObjectName("verticalLayout_KMEANS_SMOTE")
        self.nclusters_para_kmeans_smote = QtWidgets.QLabel()
        self.nclusters_para_kmeans_smote.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                                 line-height:26px '''
        )
        self.nclusters_para_kmeans_smote.setText("Number of Clusters")
        self.verticalLayout_KMEANS_SMOTE.addWidget(self.nclusters_para_kmeans_smote)
        self.num_clusters_value_kmeans_smote = QtWidgets.QSpinBox(self.page_KMEANS_SMOTE)
        self.num_clusters_value_kmeans_smote.setMinimumSize(QtCore.QSize(140, 35))
        self.num_clusters_value_kmeans_smote.setMaximumSize(QtCore.QSize(140, 35))
        self.num_clusters_value_kmeans_smote.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.num_clusters_value_kmeans_smote.setAlignment(QtCore.Qt.AlignCenter)
        self.num_clusters_value_kmeans_smote.setMaximum(1000000)
        self.verticalLayout_KMEANS_SMOTE.addWidget(self.num_clusters_value_kmeans_smote)
        spacerItem_13 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_KMEANS_SMOTE.addItem(spacerItem_13)
        self.HLayout_KMEANS_SMOTE.addLayout(self.verticalLayout_KMEANS_SMOTE)
        self.stackedWidget.addWidget(self.page_KMEANS_SMOTE)

        self.page_lle = QtWidgets.QWidget()
        self.HLayout_lle = QtWidgets.QHBoxLayout(self.page_lle)
        self.HLayout_lle.setSpacing(12)
        self.verticalLayout_lle1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_lle1.setSpacing(0)
        self.verticalLayout_lle1.setObjectName("verticalLayout_lle1")
        self.kn_para_lle = QtWidgets.QLabel()
        self.kn_para_lle.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                         line-height:26px ''')
        self.kn_para_lle.setText("Num_Neighbours")
        self.kn_para_lle.setToolTip("Number of Neighbours")
        self.kn_para_lle.setToolTipDuration(3000)
        self.verticalLayout_lle1.addWidget(self.kn_para_lle)
        self.kn_value_lle = QtWidgets.QSpinBox(self.page_lle)
        self.kn_value_lle.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.kn_value_lle.setAlignment(QtCore.Qt.AlignCenter)
        self.kn_value_lle.setMinimumSize(QtCore.QSize(140, 35))
        self.kn_value_lle.setMaximumSize(QtCore.QSize(140, 35))
        self.kn_value_lle.setMaximum(1000000)
        self.verticalLayout_lle1.addWidget(self.kn_value_lle)
        spacerItem_14 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_lle1.addItem(spacerItem_14)
        self.HLayout_lle.addLayout(self.verticalLayout_lle1)
        self.verticalLayout_lle2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_lle2.setSpacing(0)
        self.reg_para = QtWidgets.QLabel()
        self.reg_para.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                 line-height:26px '''
        )
        self.reg_para.setText("Regularization Factor")
        self.verticalLayout_lle2.addWidget(self.reg_para)
        self.reg_value = QtWidgets.QDoubleSpinBox(self.page_lle)
        self.reg_value.setMinimumSize(QtCore.QSize(140, 35))
        self.reg_value.setMaximumSize(QtCore.QSize(140, 35))
        self.reg_value.setMaximum(100000)
        self.reg_value.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.reg_value.setAlignment(QtCore.Qt.AlignCenter)
        self.verticalLayout_lle2.addWidget(self.reg_value)
        spacerItem_15 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_lle2.addItem(spacerItem_15)
        self.HLayout_lle.addLayout(self.verticalLayout_lle2)
        self.verticalLayout_lle3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_lle3.setSpacing(0)
        self.con_para = QtWidgets.QLabel()
        self.con_para.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                         line-height:26px '''
        )
        self.con_para.setText("Num_Components")
        self.con_para.setToolTip("Number of Components")
        self.con_para.setToolTipDuration(3000)
        self.verticalLayout_lle3.addWidget(self.con_para)
        self.con_value = QtWidgets.QSpinBox(self.page_lle)
        self.con_value.setMinimumSize(QtCore.QSize(140, 35))
        self.con_value.setMaximumSize(QtCore.QSize(140, 35))
        self.con_value.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.con_value.setAlignment(QtCore.Qt.AlignCenter)
        self.con_value.setMaximum(1000000)
        self.verticalLayout_lle3.addWidget(self.con_value)
        spacerItem_16 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_lle3.addItem(spacerItem_16)
        self.HLayout_lle.addLayout(self.verticalLayout_lle3)
        self.stackedWidget.addWidget(self.page_lle)

        self.page_MIXUP = QtWidgets.QWidget()
        self.page_MIXUP.setGeometry(QtCore.QRect(0, 10, 530, 10))
        self.HLayout_MIXUP = QtWidgets.QHBoxLayout(self.page_MIXUP)
        self.HLayout_MIXUP.setSpacing(12)
        self.verticalLayout_MIXUP = QtWidgets.QVBoxLayout()
        self.verticalLayout_MIXUP.setSpacing(0)
        self.verticalLayout_MIXUP.setObjectName("verticalLayout_MIXUP")
        self.alpha_para_mixup = QtWidgets.QLabel()
        self.alpha_para_mixup.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                                 line-height:26px '''
        )
        self.alpha_para_mixup.setText("Alpha")
        self.verticalLayout_MIXUP.addWidget(self.alpha_para_mixup)
        self.alpha_value = QtWidgets.QDoubleSpinBox(self.page_MIXUP)
        self.alpha_value.setMinimumSize(QtCore.QSize(140, 35))
        self.alpha_value.setMaximumSize(QtCore.QSize(140, 35))
        self.alpha_value.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.alpha_value.setAlignment(QtCore.Qt.AlignCenter)
        self.alpha_value.setMaximum(1000000)
        self.verticalLayout_MIXUP.addWidget(self.alpha_value)
        spacerItem_13 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_MIXUP.addItem(spacerItem_13)
        self.HLayout_MIXUP.addLayout(self.verticalLayout_MIXUP)
        self.stackedWidget.addWidget(self.page_MIXUP)

        self.page_knnmtd = QtWidgets.QWidget()
        self.HLayout_knnmtd = QtWidgets.QHBoxLayout(self.page_knnmtd)
        self.HLayout_knnmtd.setSpacing(12)
        self.verticalLayout_knnmtd = QtWidgets.QVBoxLayout()
        self.verticalLayout_knnmtd.setSpacing(0)
        self.verticalLayout_knnmtd.setObjectName("verticalLayout_knnmtd")
        self.kn_para_knnmtd = QtWidgets.QLabel()
        self.kn_para_knnmtd.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                                         line-height:26px '''
        )
        self.kn_para_knnmtd.setText("Number of Neighbours")

        self.verticalLayout_knnmtd.addWidget(self.kn_para_knnmtd)
        self.kn_value_knnmtd = QtWidgets.QSpinBox(self.page_knnmtd)
        self.kn_value_knnmtd.setMinimumSize(QtCore.QSize(140, 35))
        self.kn_value_knnmtd.setMaximumSize(QtCore.QSize(140, 35))
        self.kn_value_knnmtd.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.kn_value_knnmtd.setAlignment(QtCore.Qt.AlignCenter)
        self.kn_value_knnmtd.setMaximum(1000000)
        self.verticalLayout_knnmtd.addWidget(self.kn_value_knnmtd)
        spacerItem_17 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_knnmtd.addItem(spacerItem_17)
        self.HLayout_knnmtd.addLayout(self.verticalLayout_knnmtd)
        self.stackedWidget.addWidget(self.page_knnmtd)

        self.page_gmm = QtWidgets.QWidget()
        self.HLayout_gmm = QtWidgets.QHBoxLayout(self.page_gmm)
        self.HLayout_gmm.setSpacing(12)
        self.verticalLayout_gmm = QtWidgets.QVBoxLayout()
        self.verticalLayout_gmm.setSpacing(0)
        self.verticalLayout_gmm.setObjectName("verticalLayout_gmm")
        self.con_para_gmm = QtWidgets.QLabel()
        self.con_para_gmm.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#898989;
                                                                                line-height:26px '''
        )
        self.con_para_gmm.setText("Number of components")

        self.verticalLayout_gmm.addWidget(self.con_para_gmm)
        self.con_value_gmm = QtWidgets.QSpinBox(self.page_gmm)
        self.con_value_gmm.setMinimumSize(QtCore.QSize(140, 35))
        self.con_value_gmm.setMaximumSize(QtCore.QSize(140, 35))
        self.con_value_gmm.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.con_value_gmm.setAlignment(QtCore.Qt.AlignCenter)
        self.con_value_gmm.setMaximum(1000000)
        self.verticalLayout_gmm.addWidget(self.con_value_gmm)
        spacerItem_18 = QtWidgets.QSpacerItem(280, 30,
                                              QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_gmm.addItem(spacerItem_18)
        self.HLayout_gmm.addLayout(self.verticalLayout_gmm)
        self.stackedWidget.addWidget(self.page_gmm)

        self.page_gan = QtWidgets.QWidget()
        self.VLayout_gan = QtWidgets.QVBoxLayout(self.page_gan)
        self.VLayout_gan.setSpacing(9)
        self.HLayout_gan1 = QtWidgets.QHBoxLayout(self.page_gan)
        self.HLayout_gan1.setSpacing(15)
        self.verticalLayout_gan1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_gan1.setSpacing(8)
        self.verticalLayout_gan1.setObjectName("verticalLayout_gan1")
        self.epoch_para_gan = QtWidgets.QLabel()
        self.epoch_para_gan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                         line-height:26px '''
        )
        self.epoch_para_gan.setText("Number of Epochs")
        self.verticalLayout_gan1.addWidget(self.epoch_para_gan)
        self.epoch_value_gan_vae = QtWidgets.QSpinBox(self.page_gan)
        self.epoch_value_gan_vae.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.epoch_value_gan_vae.setAlignment(QtCore.Qt.AlignCenter)
        self.epoch_value_gan_vae.setMaximum(1000000)
        self.epoch_value_gan_vae.setMinimumSize(QtCore.QSize(140, 35))
        self.epoch_value_gan_vae.setMaximumSize(QtCore.QSize(140, 35))
        self.verticalLayout_gan1.addWidget(self.epoch_value_gan_vae)

        self.HLayout_gan1.addLayout(self.verticalLayout_gan1)
        self.verticalLayout_gan2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_gan2.setSpacing(8)
        self.lr_para_gan = QtWidgets.QLabel()
        self.lr_para_gan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.lr_para_gan.setText("Learning Rate")
        self.verticalLayout_gan2.addWidget(self.lr_para_gan)
        self.lr_value_gan_vae = QtWidgets.QDoubleSpinBox(self.page_gan)
        self.lr_value_gan_vae.setMinimumSize(QtCore.QSize(140, 35))
        self.lr_value_gan_vae.setMaximumSize(QtCore.QSize(140, 35))
        self.lr_value_gan_vae.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.lr_value_gan_vae.setAlignment(QtCore.Qt.AlignCenter)
        self.lr_value_gan_vae.setMaximum(1)
        self.lr_value_gan_vae.setDecimals(5)
        self.verticalLayout_gan2.addWidget(self.lr_value_gan_vae)
        self.HLayout_gan1.addLayout(self.verticalLayout_gan2)

        self.verticalLayout_gan3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_gan3.setSpacing(8)
        self.batch_para_gan = QtWidgets.QLabel()
        self.batch_para_gan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                         line-height:26px '''
        )
        self.batch_para_gan.setText("Batch Size")
        self.verticalLayout_gan3.addWidget(self.batch_para_gan)
        self.batch_value_gan_vae = QtWidgets.QSpinBox(self.page_gan)
        self.batch_value_gan_vae.setMinimumSize(QtCore.QSize(140, 35))
        self.batch_value_gan_vae.setMaximumSize(QtCore.QSize(140, 35))
        self.batch_value_gan_vae.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.batch_value_gan_vae.setAlignment(QtCore.Qt.AlignCenter)
        self.batch_value_gan_vae.setMaximum(1000)
        self.verticalLayout_gan3.addWidget(self.batch_value_gan_vae)
        self.HLayout_gan1.addLayout(self.verticalLayout_gan3)
        self.HLayout_gan2 = QtWidgets.QHBoxLayout(self.page_gan)
        self.HLayout_gan2.setSpacing(12)
        self.verticalLayout_gan4 = QtWidgets.QVBoxLayout(self.page_gan)
        self.verticalLayout_gan4.setSpacing(8)
        self.verticalLayout_gan4.setObjectName("verticalLayout_gan4")
        self.latent_para_gan = QtWidgets.QLabel()
        self.latent_para_gan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.latent_para_gan.setText("Latent Dimension")
        self.verticalLayout_gan4.addWidget(self.latent_para_gan)
        self.latent_value_gan_vae = QtWidgets.QSpinBox(self.page_gan)
        self.latent_value_gan_vae.setStyleSheet(
            '''border:1px solid #CCCCCC ''')
        self.latent_value_gan_vae.setAlignment(QtCore.Qt.AlignCenter)
        self.latent_value_gan_vae.setMinimumSize(QtCore.QSize(140, 35))
        self.latent_value_gan_vae.setMaximumSize(QtCore.QSize(140, 35))
        self.latent_value_gan_vae.setMaximum(1000)
        self.verticalLayout_gan4.addWidget(self.latent_value_gan_vae)
        self.HLayout_gan2.addLayout(self.verticalLayout_gan4)
        self.VLayout_gan.addLayout(self.HLayout_gan1)
        self.VLayout_gan.addLayout(self.HLayout_gan2)
        self.stackedWidget.addWidget(self.page_gan)

        self.page_wgan_gp = QtWidgets.QWidget()
        self.VLayout_wgan_gp = QtWidgets.QVBoxLayout(self.page_wgan_gp)
        self.VLayout_wgan_gp.setSpacing(9)
        self.HLayout_wgan_gp1 = QtWidgets.QHBoxLayout(self.page_wgan_gp)
        self.HLayout_wgan_gp1.setSpacing(15)
        self.verticalLayout_wgan_gp1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_wgan_gp1.setSpacing(8)
        self.verticalLayout_wgan_gp1.setObjectName("verticalLayout_gan1")
        self.iter_para_wgan_gp = QtWidgets.QLabel()
        self.iter_para_wgan_gp.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                         line-height:26px '''
        )
        self.iter_para_wgan_gp.setText("Number of Iters")
        self.verticalLayout_wgan_gp1.addWidget(self.iter_para_wgan_gp)
        self.iter_value_wgan_gp = QtWidgets.QSpinBox(self.page_wgan_gp)
        self.iter_value_wgan_gp.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.iter_value_wgan_gp.setAlignment(QtCore.Qt.AlignCenter)
        self.iter_value_wgan_gp.setMaximum(1000000)
        self.iter_value_wgan_gp.setMinimumSize(QtCore.QSize(140, 35))
        self.iter_value_wgan_gp.setMaximumSize(QtCore.QSize(140, 35))
        self.verticalLayout_wgan_gp1.addWidget(self.iter_value_wgan_gp)

        self.HLayout_wgan_gp1.addLayout(self.verticalLayout_wgan_gp1)
        self.verticalLayout_wgan_gp2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_wgan_gp2.setSpacing(8)
        self.lr_para_wgan_gp = QtWidgets.QLabel()
        self.lr_para_wgan_gp.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.lr_para_wgan_gp.setText("Learning Rate")
        self.verticalLayout_wgan_gp2.addWidget(self.lr_para_wgan_gp)
        self.lr_value_wgan_gp = QtWidgets.QDoubleSpinBox(self.page_wgan_gp)
        self.lr_value_wgan_gp.setMinimumSize(QtCore.QSize(140, 35))
        self.lr_value_wgan_gp.setMaximumSize(QtCore.QSize(140, 35))
        self.lr_value_wgan_gp.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.lr_value_wgan_gp.setAlignment(QtCore.Qt.AlignCenter)
        self.lr_value_wgan_gp.setMaximum(1)
        self.lr_value_wgan_gp.setDecimals(5)
        self.verticalLayout_wgan_gp2.addWidget(self.lr_value_wgan_gp)
        self.HLayout_wgan_gp1.addLayout(self.verticalLayout_wgan_gp2)

        self.verticalLayout_wgan_gp3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_wgan_gp3.setSpacing(8)
        self.batch_para_wgan_gp = QtWidgets.QLabel()
        self.batch_para_wgan_gp.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                         line-height:26px '''
        )
        self.batch_para_wgan_gp.setText("Batch Size")
        self.verticalLayout_wgan_gp3.addWidget(self.batch_para_wgan_gp)
        self.batch_value_wgan_gp = QtWidgets.QSpinBox(self.page_wgan_gp)
        self.batch_value_wgan_gp.setMinimumSize(QtCore.QSize(140, 35))
        self.batch_value_wgan_gp.setMaximumSize(QtCore.QSize(140, 35))
        self.batch_value_wgan_gp.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.batch_value_wgan_gp.setAlignment(QtCore.Qt.AlignCenter)
        self.batch_value_wgan_gp.setMaximum(1000)
        self.verticalLayout_wgan_gp3.addWidget(self.batch_value_wgan_gp)
        self.HLayout_wgan_gp1.addLayout(self.verticalLayout_wgan_gp3)
        self.HLayout_wgan_gp2 = QtWidgets.QHBoxLayout(self.page_wgan_gp)
        self.HLayout_wgan_gp2.setSpacing(12)
        self.verticalLayout_wgan_gp4 = QtWidgets.QVBoxLayout(self.page_wgan_gp)
        self.verticalLayout_wgan_gp4.setSpacing(8)
        self.verticalLayout_wgan_gp4.setObjectName("verticalLayout_gan4")
        self.latent_para_wgan_gp = QtWidgets.QLabel()
        self.latent_para_wgan_gp.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.latent_para_wgan_gp.setText("Latent Dimension")
        self.verticalLayout_wgan_gp4.addWidget(self.latent_para_wgan_gp)
        self.latent_value_wgan_gp = QtWidgets.QSpinBox(self.page_wgan_gp)
        self.latent_value_wgan_gp.setStyleSheet(
            '''border:1px solid #CCCCCC ''')
        self.latent_value_wgan_gp.setAlignment(QtCore.Qt.AlignCenter)
        self.latent_value_wgan_gp.setMinimumSize(QtCore.QSize(140, 35))
        self.latent_value_wgan_gp.setMaximumSize(QtCore.QSize(140, 35))
        self.latent_value_wgan_gp.setMaximum(1000)
        self.verticalLayout_wgan_gp4.addWidget(self.latent_value_wgan_gp)
        self.HLayout_wgan_gp2.addLayout(self.verticalLayout_wgan_gp4)

        self.verticalLayout_wgan_gp5 = QtWidgets.QVBoxLayout(self.page_wgan_gp)
        self.verticalLayout_wgan_gp5.setSpacing(8)
        self.n_critic_para_wgan_gp = QtWidgets.QLabel()
        self.n_critic_para_wgan_gp.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.n_critic_para_wgan_gp.setText("Critic Iters")
        self.verticalLayout_wgan_gp5.addWidget(self.n_critic_para_wgan_gp)
        self.n_critic_value_wgan_gp = QtWidgets.QSpinBox(self.page_wgan_gp)
        self.n_critic_value_wgan_gp.setStyleSheet(
            '''border:1px solid #CCCCCC ''')
        self.n_critic_value_wgan_gp.setAlignment(QtCore.Qt.AlignCenter)
        self.n_critic_value_wgan_gp.setMinimumSize(QtCore.QSize(140, 35))
        self.n_critic_value_wgan_gp.setMaximumSize(QtCore.QSize(140, 35))
        self.n_critic_value_wgan_gp.setMaximum(100)
        self.verticalLayout_wgan_gp5.addWidget(self.n_critic_value_wgan_gp)
        self.HLayout_wgan_gp2.addLayout(self.verticalLayout_wgan_gp5)
        self.verticalLayout_wgan_gp6 = QtWidgets.QVBoxLayout(self.page_wgan_gp)
        self.verticalLayout_wgan_gp6.setSpacing(8)
        self.lambda_para_wgan_gp = QtWidgets.QLabel()
        self.lambda_para_wgan_gp.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.lambda_para_wgan_gp.setText("Lambda")
        self.verticalLayout_wgan_gp6.addWidget(self.lambda_para_wgan_gp)
        self.lambda_value_wgan_gp = QtWidgets.QSpinBox(self.page_wgan_gp)
        self.lambda_value_wgan_gp.setStyleSheet(
            '''border:1px solid #CCCCCC ''')
        self.lambda_value_wgan_gp.setAlignment(QtCore.Qt.AlignCenter)
        self.lambda_value_wgan_gp.setMinimumSize(QtCore.QSize(140, 35))
        self.lambda_value_wgan_gp.setMaximumSize(QtCore.QSize(140, 35))
        self.lambda_value_wgan_gp.setMaximum(100)
        self.verticalLayout_wgan_gp6.addWidget(self.lambda_value_wgan_gp)
        self.HLayout_wgan_gp2.addLayout(self.verticalLayout_wgan_gp6)
        self.VLayout_wgan_gp.addLayout(self.HLayout_wgan_gp1)
        self.VLayout_wgan_gp.addLayout(self.HLayout_wgan_gp2)
        self.stackedWidget.addWidget(self.page_wgan_gp)

        self.page_lsgan = QtWidgets.QWidget()
        self.VLayout_lsgan = QtWidgets.QVBoxLayout(self.page_lsgan)
        self.VLayout_lsgan.setSpacing(9)
        self.HLayout_lsgan1 = QtWidgets.QHBoxLayout(self.page_lsgan)
        self.HLayout_lsgan1.setSpacing(15)
        self.verticalLayout_lsgan1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_lsgan1.setSpacing(8)
        self.verticalLayout_lsgan1.setObjectName("verticalLayout_lsgan1")
        self.iter_para_lsgan = QtWidgets.QLabel()
        self.iter_para_lsgan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                         line-height:26px '''
        )
        self.iter_para_lsgan.setText("Number of Iters")
        self.verticalLayout_lsgan1.addWidget(self.iter_para_lsgan)
        self.iter_value_lsgan = QtWidgets.QSpinBox(self.page_lsgan)
        self.iter_value_lsgan.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.iter_value_lsgan.setAlignment(QtCore.Qt.AlignCenter)
        self.iter_value_lsgan.setMaximum(1000000)
        self.iter_value_lsgan.setMinimumSize(QtCore.QSize(140, 35))
        self.iter_value_lsgan.setMaximumSize(QtCore.QSize(140, 35))
        self.verticalLayout_lsgan1.addWidget(self.iter_value_lsgan)

        self.HLayout_lsgan1.addLayout(self.verticalLayout_lsgan1)
        self.verticalLayout_lsgan2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_lsgan2.setSpacing(8)
        self.lr_para_lsgan = QtWidgets.QLabel()
        self.lr_para_lsgan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.lr_para_lsgan.setText("Learning Rate")
        self.verticalLayout_lsgan2.addWidget(self.lr_para_lsgan)
        self.lr_value_lsgan = QtWidgets.QDoubleSpinBox(self.page_lsgan)
        self.lr_value_lsgan.setMinimumSize(QtCore.QSize(140, 35))
        self.lr_value_lsgan.setMaximumSize(QtCore.QSize(140, 35))
        self.lr_value_lsgan.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.lr_value_lsgan.setAlignment(QtCore.Qt.AlignCenter)
        self.lr_value_lsgan.setMaximum(1)
        self.lr_value_lsgan.setDecimals(5)
        self.verticalLayout_lsgan2.addWidget(self.lr_value_lsgan)
        self.HLayout_lsgan1.addLayout(self.verticalLayout_lsgan2)

        self.verticalLayout_lsgan3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_lsgan3.setSpacing(8)
        self.batch_para_lsgan = QtWidgets.QLabel()
        self.batch_para_lsgan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                         line-height:26px '''
        )
        self.batch_para_lsgan.setText("Batch Size")
        self.verticalLayout_lsgan3.addWidget(self.batch_para_lsgan)
        self.batch_value_lsgan = QtWidgets.QSpinBox(self.page_lsgan)
        self.batch_value_lsgan.setMinimumSize(QtCore.QSize(140, 35))
        self.batch_value_lsgan.setMaximumSize(QtCore.QSize(140, 35))
        self.batch_value_lsgan.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.batch_value_lsgan.setAlignment(QtCore.Qt.AlignCenter)
        self.batch_value_lsgan.setMaximum(1000)
        self.verticalLayout_lsgan3.addWidget(self.batch_value_lsgan)
        self.HLayout_lsgan1.addLayout(self.verticalLayout_lsgan3)
        self.HLayout_lsgan2 = QtWidgets.QHBoxLayout(self.page_lsgan)
        self.HLayout_lsgan2.setSpacing(12)
        self.verticalLayout_lsgan4 = QtWidgets.QVBoxLayout(self.page_lsgan)
        self.verticalLayout_lsgan4.setSpacing(8)
        self.verticalLayout_lsgan4.setObjectName("verticalLayout_gan4")
        self.latent_para_lsgan = QtWidgets.QLabel()
        self.latent_para_lsgan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.latent_para_lsgan.setText("Latent Dimension")
        self.verticalLayout_lsgan4.addWidget(self.latent_para_lsgan)
        self.latent_value_lsgan = QtWidgets.QSpinBox(self.page_lsgan)
        self.latent_value_lsgan.setStyleSheet(
            '''border:1px solid #CCCCCC ''')
        self.latent_value_lsgan.setAlignment(QtCore.Qt.AlignCenter)
        self.latent_value_lsgan.setMinimumSize(QtCore.QSize(140, 35))
        self.latent_value_lsgan.setMaximumSize(QtCore.QSize(140, 35))
        self.latent_value_lsgan.setMaximum(1000)
        self.verticalLayout_lsgan4.addWidget(self.latent_value_lsgan)
        self.HLayout_lsgan2.addLayout(self.verticalLayout_lsgan4)

        self.verticalLayout_lsgan5 = QtWidgets.QVBoxLayout(self.page_lsgan)
        self.verticalLayout_lsgan5.setSpacing(8)
        self.n_critic_para_lsgan = QtWidgets.QLabel()
        self.n_critic_para_lsgan.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.n_critic_para_lsgan.setText("Critic Iters")
        self.verticalLayout_lsgan5.addWidget(self.n_critic_para_lsgan)
        self.n_critic_value_lsgan = QtWidgets.QSpinBox(self.page_lsgan)
        self.n_critic_value_lsgan.setStyleSheet(
            '''border:1px solid #CCCCCC ''')
        self.n_critic_value_lsgan.setAlignment(QtCore.Qt.AlignCenter)
        self.n_critic_value_lsgan.setMinimumSize(QtCore.QSize(140, 35))
        self.n_critic_value_lsgan.setMaximumSize(QtCore.QSize(140, 35))
        self.n_critic_value_lsgan.setMaximum(100)
        self.verticalLayout_lsgan5.addWidget(self.n_critic_value_lsgan)
        self.HLayout_lsgan2.addLayout(self.verticalLayout_lsgan5)
        self.VLayout_lsgan.addLayout(self.HLayout_lsgan1)
        self.VLayout_lsgan.addLayout(self.HLayout_lsgan2)
        self.stackedWidget.addWidget(self.page_lsgan)

        self.page_flow = QtWidgets.QWidget()
        self.VLayout_flow = QtWidgets.QVBoxLayout(self.page_flow)
        self.VLayout_flow.setSpacing(9)
        self.HLayout_flow1 = QtWidgets.QHBoxLayout(self.page_flow)
        self.HLayout_flow1.setSpacing(15)
        self.verticalLayout_flow1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_flow1.setSpacing(8)
        self.epoch_para_flow = QtWidgets.QLabel()
        self.epoch_para_flow.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                         line-height:26px '''
        )
        self.epoch_para_flow.setText("Number of Epochs")
        self.verticalLayout_flow1.addWidget(self.epoch_para_flow)
        self.epoch_value_flow = QtWidgets.QSpinBox(self.page_flow)
        self.epoch_value_flow.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.epoch_value_flow.setAlignment(QtCore.Qt.AlignCenter)
        self.epoch_value_flow.setMaximum(1000000)
        self.epoch_value_flow.setMinimumSize(QtCore.QSize(140, 35))
        self.epoch_value_flow.setMaximumSize(QtCore.QSize(140, 35))
        self.verticalLayout_flow1.addWidget(self.epoch_value_flow)

        self.HLayout_flow1.addLayout(self.verticalLayout_flow1)
        self.verticalLayout_flow2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_flow2.setSpacing(8)
        self.lr_para_flow = QtWidgets.QLabel()
        self.lr_para_flow.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.lr_para_flow.setText("Learning Rate")
        self.verticalLayout_flow2.addWidget(self.lr_para_flow)
        self.lr_value_flow = QtWidgets.QDoubleSpinBox(self.page_flow)
        self.lr_value_flow.setMinimumSize(QtCore.QSize(140, 35))
        self.lr_value_flow.setMaximumSize(QtCore.QSize(140, 35))
        self.lr_value_flow.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.lr_value_flow.setAlignment(QtCore.Qt.AlignCenter)
        self.lr_value_flow.setMaximum(1)
        self.lr_value_flow.setDecimals(5)
        self.verticalLayout_flow2.addWidget(self.lr_value_flow)
        self.HLayout_flow1.addLayout(self.verticalLayout_flow2)

        self.verticalLayout_flow3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_flow3.setSpacing(8)
        self.batch_para_flow = QtWidgets.QLabel()
        self.batch_para_flow.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                         line-height:26px '''
        )
        self.batch_para_flow.setText("Batch Size")
        self.verticalLayout_flow3.addWidget(self.batch_para_flow)
        self.batch_value_flow = QtWidgets.QSpinBox(self.page_flow)
        self.batch_value_flow.setMinimumSize(QtCore.QSize(140, 35))
        self.batch_value_flow.setMaximumSize(QtCore.QSize(140, 35))
        self.batch_value_flow.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.batch_value_flow.setAlignment(QtCore.Qt.AlignCenter)
        self.batch_value_flow.setMaximum(1000)
        self.verticalLayout_flow3.addWidget(self.batch_value_flow)
        self.HLayout_flow1.addLayout(self.verticalLayout_flow3)
        self.HLayout_flow2 = QtWidgets.QHBoxLayout(self.page_flow)
        self.HLayout_flow2.setSpacing(12)
        self.verticalLayout_flow4 = QtWidgets.QVBoxLayout(self.page_flow)
        self.verticalLayout_flow4.setSpacing(8)
        self.verticalLayout_flow4.setObjectName("verticalLayout_gan4")
        self.num_blocks_para_flow = QtWidgets.QLabel()
        self.num_blocks_para_flow.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.num_blocks_para_flow.setText("Number of Blocks")
        self.verticalLayout_flow4.addWidget(self.num_blocks_para_flow)
        self.num_blocks_value_flow = QtWidgets.QSpinBox(self.page_gan)
        self.num_blocks_value_flow.setStyleSheet(
            '''border:1px solid #CCCCCC ''')
        self.num_blocks_value_flow.setAlignment(QtCore.Qt.AlignCenter)
        self.num_blocks_value_flow.setMinimumSize(QtCore.QSize(140, 35))
        self.num_blocks_value_flow.setMaximumSize(QtCore.QSize(140, 35))
        self.num_blocks_value_flow.setMaximum(1000)
        self.verticalLayout_flow4.addWidget(self.num_blocks_value_flow)
        self.HLayout_flow2.addLayout(self.verticalLayout_flow4)
        self.VLayout_flow.addLayout(self.HLayout_flow1)
        self.VLayout_flow.addLayout(self.HLayout_flow2)
        self.stackedWidget.addWidget(self.page_flow)

        self.page_ddpm = QtWidgets.QWidget()
        self.VLayout_ddpm = QtWidgets.QVBoxLayout(self.page_ddpm)
        self.VLayout_ddpm.setSpacing(9)
        self.HLayout_ddpm1 = QtWidgets.QHBoxLayout(self.page_ddpm)
        self.HLayout_ddpm1.setSpacing(15)
        self.verticalLayout_ddpm1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_ddpm1.setSpacing(8)
        self.epoch_para_ddpm = QtWidgets.QLabel()
        self.epoch_para_ddpm.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                         line-height:26px '''
        )
        self.epoch_para_ddpm.setText("Number of Epochs")
        self.verticalLayout_ddpm1.addWidget(self.epoch_para_ddpm)
        self.epoch_value_ddpm = QtWidgets.QSpinBox(self.page_ddpm)
        self.epoch_value_ddpm.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.epoch_value_ddpm.setAlignment(QtCore.Qt.AlignCenter)
        self.epoch_value_ddpm.setMaximum(1000000)
        self.epoch_value_ddpm.setMinimumSize(QtCore.QSize(140, 35))
        self.epoch_value_ddpm.setMaximumSize(QtCore.QSize(140, 35))
        self.verticalLayout_ddpm1.addWidget(self.epoch_value_ddpm)

        self.HLayout_ddpm1.addLayout(self.verticalLayout_ddpm1)
        self.verticalLayout_ddpm2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_ddpm2.setSpacing(8)
        self.lr_para_ddpm = QtWidgets.QLabel()
        self.lr_para_ddpm.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.lr_para_ddpm.setText("Learning Rate")
        self.verticalLayout_ddpm2.addWidget(self.lr_para_ddpm)
        self.lr_value_ddpm = QtWidgets.QDoubleSpinBox(self.page_ddpm)
        self.lr_value_ddpm.setMinimumSize(QtCore.QSize(140, 35))
        self.lr_value_ddpm.setMaximumSize(QtCore.QSize(140, 35))
        self.lr_value_ddpm.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.lr_value_ddpm.setAlignment(QtCore.Qt.AlignCenter)
        self.lr_value_ddpm.setMaximum(1)
        self.lr_value_ddpm.setDecimals(5)
        self.verticalLayout_ddpm2.addWidget(self.lr_value_ddpm)
        self.HLayout_ddpm1.addLayout(self.verticalLayout_ddpm2)

        self.verticalLayout_ddpm3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_ddpm3.setSpacing(8)
        self.batch_para_ddpm = QtWidgets.QLabel()
        self.batch_para_ddpm.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                         line-height:26px '''
        )
        self.batch_para_ddpm.setText("Batch Size")
        self.verticalLayout_ddpm3.addWidget(self.batch_para_ddpm)
        self.batch_value_ddpm = QtWidgets.QSpinBox(self.page_ddpm)
        self.batch_value_ddpm.setMinimumSize(QtCore.QSize(140, 35))
        self.batch_value_ddpm.setMaximumSize(QtCore.QSize(140, 35))
        self.batch_value_ddpm.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.batch_value_ddpm.setAlignment(QtCore.Qt.AlignCenter)
        self.batch_value_ddpm.setMaximum(1000)
        self.verticalLayout_ddpm3.addWidget(self.batch_value_ddpm)
        self.HLayout_ddpm1.addLayout(self.verticalLayout_ddpm3)
        self.HLayout_ddpm2 = QtWidgets.QHBoxLayout(self.page_ddpm)
        self.HLayout_ddpm2.setSpacing(12)
        self.verticalLayout_ddpm4 = QtWidgets.QVBoxLayout(self.page_ddpm)
        self.verticalLayout_ddpm4.setSpacing(8)
        self.verticalLayout_ddpm4.setObjectName("verticalLayout_gan4")
        self.num_steps_para_ddpm = QtWidgets.QLabel()
        self.num_steps_para_ddpm.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                 line-height:26px '''
        )
        self.num_steps_para_ddpm.setText("Number of Time Steps")
        self.verticalLayout_ddpm4.addWidget(self.num_steps_para_ddpm)
        self.num_steps_value_ddpm = QtWidgets.QSpinBox(self.page_gan)
        self.num_steps_value_ddpm.setStyleSheet(
            '''border:1px solid #CCCCCC ''')
        self.num_steps_value_ddpm.setAlignment(QtCore.Qt.AlignCenter)
        self.num_steps_value_ddpm.setMinimumSize(QtCore.QSize(140, 35))
        self.num_steps_value_ddpm.setMaximumSize(QtCore.QSize(140, 35))
        self.num_steps_value_ddpm.setMaximum(1000)
        self.verticalLayout_ddpm4.addWidget(self.num_steps_value_ddpm)
        self.HLayout_ddpm2.addLayout(self.verticalLayout_ddpm4)
        self.VLayout_ddpm.addLayout(self.HLayout_ddpm1)
        self.VLayout_ddpm.addLayout(self.HLayout_ddpm2)
        self.stackedWidget.addWidget(self.page_ddpm)

        self.widget_42 = QtWidgets.QWidget(Form)
        self.widget_42.setObjectName("widget_42")
        self.widget_42.setMaximumSize(QtCore.QSize(167755555, 11111111))
        self.widget_42.setMinimumSize(QtCore.QSize(534, 502))
        self.widget_generation = QtWidgets.QWidget(self.widget_42)
        self.widget_generation.setObjectName("widget_para")
        self.widget_generation.setGeometry(QtCore.QRect(15, 10, 184, 30))
        self.horizontalLayout_42 = QtWidgets.QHBoxLayout(
            self.widget_generation)
        self.horizontalLayout_42.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_42.setSpacing(9)
        self.horizontalLayout_42.setObjectName("horizontalLayout_42")
        self.label_9 = QtWidgets.QLabel(self.widget_generation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setMaximumSize(QtCore.QSize(14, 14))
        self.label_9.setMinimumSize(QtCore.QSize(14, 14))
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_42.addWidget(self.label_9)
        self.label_10 = QtWidgets.QLabel(self.widget_generation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_42.addWidget(self.label_10)
        self.widget_numpara = QtWidgets.QWidget(Form)
        self.widget_numpara.setObjectName("widget_numpara")
        self.widget_numpara.setGeometry(QtCore.QRect(257, 322, 534, 40))
        self.horizontalLayout_43 = QtWidgets.QHBoxLayout(self.widget_numpara)
        self.horizontalLayout_43.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_43.setSpacing(0)
        self.horizontalLayout_43.setObjectName("horizontalLayout_43")
        self.numgen_para = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.numgen_para.sizePolicy().hasHeightForWidth())
        self.numgen_para.setSizePolicy(sizePolicy)
        self.numgen_para.setObjectName("numgen_para")
        self.numgen_para.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#898989;
                                                                                line-height:26px '''
        )
        self.numgen_para.setText("Number of Generated Samples")
        self.num_gen_value = QtWidgets.QSpinBox()
        self.num_gen_value.setMinimumSize(QtCore.QSize(140, 30))
        self.num_gen_value.setMaximumSize(QtCore.QSize(140, 30))
        self.num_gen_value.setMaximum(1000000)
        self.num_gen_value.setValue(1000)
        self.num_gen_value.setValue(1000)
        self.num_gen_value.setStyleSheet('''border:1px solid #CCCCCC ''')
        self.num_gen_value.setAlignment(QtCore.Qt.AlignCenter)
        self.pushButton_start = QtWidgets.QPushButton()
        self.pushButton_start.setObjectName("pushButton_start")
        self.pushButton_start.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_start.setMaximumSize(QtCore.QSize(100, 40))
        self.pushButton_start.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;font-weight:500;font-size:16px}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_start.setText("Start")
        self.pushButton_start.setFont(font)

        spacerItem11 = QtWidgets.QSpacerItem(20, 20,
                                             QtWidgets.QSizePolicy.Maximum,
                                             QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_43.addWidget(self.numgen_para)
        self.horizontalLayout_43.addWidget(self.num_gen_value)
        self.horizontalLayout_43.addItem(spacerItem11)
        self.horizontalLayout_43.addWidget(self.pushButton_start)
        self.widget_pca = QtWidgets.QWidget(Form)
        self.widget_pca.setObjectName("widget_pca")
        self.widget_pca.setGeometry(QtCore.QRect(260, 365, 534, 1000))
        self.PCADisplayGB = QtWidgets.QGroupBox(self.widget_pca)
        self.PCADisplayGB.setStyleSheet('''border:1px solid #FFFFFF ''')
        self.fig_pca = plt.Figure(figsize=(5, 3))
        self.pca_canva = FigureCanvas(self.fig_pca)
        self.axes_pca = self.pca_canva.figure.subplots()
        self.fig_pca.subplots_adjust(top=0.985,
                                     bottom=0.085,
                                     left=0.090,
                                     right=0.975,
                                     hspace=0.2,
                                     wspace=0.2)
        self.toolbar_pca = NavigationToolbar(self.pca_canva, self)
        self.layout_visual = QtWidgets.QVBoxLayout()
        self.layout_visual.addWidget(self.pca_canva)
        self.layout_visual.addWidget(self.toolbar_pca)
        self.PCADisplayGB.setLayout(self.layout_visual)
        self.widget_save = QtWidgets.QWidget(Form)
        self.widget_save.setObjectName("widget_save")
        self.widget_save.setGeometry(QtCore.QRect(257, 732, 534, 45))
        self.horizontalLayout_44 = QtWidgets.QHBoxLayout(self.widget_save)
        self.horizontalLayout_44.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_44.setSpacing(0)
        self.horizontalLayout_44.setObjectName("horizontalLayout_44")
        self.pushButton_savedata = QtWidgets.QPushButton()
        self.pushButton_savedata.setObjectName("pushButton_savedata")
        self.pushButton_savedata.setMinimumSize(QtCore.QSize(200, 40))
        self.pushButton_savedata.setMaximumSize(QtCore.QSize(200, 40))
        self.pushButton_savedata.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;font-weight:500;font-size:16px}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_savedata.setText("Save Data")
        self.pushButton_savedata.setFont(font)
        self.pushButton_savemodel = QtWidgets.QPushButton()
        self.pushButton_savemodel.setObjectName("pushButton_savemodel")
        self.pushButton_savemodel.setMinimumSize(QtCore.QSize(200, 40))
        self.pushButton_savemodel.setMaximumSize(QtCore.QSize(200, 40))
        self.pushButton_savemodel.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;font-weight:500;font-size:16px}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_savemodel.setText("Save Model")
        self.pushButton_savemodel.setFont(font)
        self.horizontalLayout_44.addWidget(self.pushButton_savedata)
        self.horizontalLayout_44.addWidget(self.pushButton_savemodel)
        self.verticalLayout_2.addWidget(self.widget_41)
        self.verticalLayout_2.addWidget(self.widget_42)
        self.horizontalLayout_2.addWidget(self.widget_4)
        self.widget_5 = QtWidgets.QWidget(Form)
        self.widget_5.setObjectName("widget_5")
        self.widget_5.setMaximumSize(QtCore.QSize(1677, 11111111))
        self.widget_5.setMinimumSize(QtCore.QSize(462, 692))
        self.widget_test = QtWidgets.QWidget(self.widget_5)
        self.widget_test.setObjectName("widget_para")
        self.widget_test.setGeometry(QtCore.QRect(15, 10, 134, 30))
        self.horizontalLayout_45 = QtWidgets.QHBoxLayout(self.widget_test)
        self.horizontalLayout_45.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_45.setSpacing(8)
        self.horizontalLayout_45.setObjectName("horizontalLayout_45")
        self.label_11 = QtWidgets.QLabel(self.widget_test)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(
            self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMaximumSize(QtCore.QSize(14, 14))
        self.label_11.setMinimumSize(QtCore.QSize(14, 14))
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_45.addWidget(self.label_11)
        self.label_12 = QtWidgets.QLabel(self.widget_test)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_45.addWidget(self.label_12)

        self.widget_model_radio = QtWidgets.QWidget(self.widget_5)
        self.widget_model_radio.setGeometry(QtCore.QRect(20, 40, 500, 400))
        self.rb1 = QtWidgets.QRadioButton(self.widget_model_radio)
        self.rb1.move(15, 8)
        self.rb1.setText("SVM")
        self.rb1.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#1f77b4;
                                                                                 line-height:26px '''
        )
        self.rb2 = QtWidgets.QRadioButton(self.widget_model_radio)
        self.rb2.move(135, 8)
        self.rb2.setText("KNN")
        self.rb2.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#1f77b4;
                                                                                 line-height:26px '''
        )
        self.rb3 = QtWidgets.QRadioButton(self.widget_model_radio)
        self.rb3.move(263, 8)
        self.rb3.setText("MLP")
        self.rb3.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#1f77b4;
                                                                                 line-height:26px '''
        )
        self.rb4 = QtWidgets.QRadioButton(self.widget_model_radio)
        self.rb4.move(15, 34)
        self.rb4.setText("DecisionTree")
        self.rb4.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#1f77b4;
                                                                                 line-height:26px '''
        )
        self.rb5 = QtWidgets.QRadioButton(self.widget_model_radio)
        self.rb5.move(135, 34)
        self.rb5.setText("RadomForest")
        self.rb5.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#1f77b4;
                                                                                 line-height:26px '''
        )
        self.rb6 = QtWidgets.QRadioButton(self.widget_model_radio)
        self.rb6.move(263, 34)
        self.rb6.setText("AdaBoost")
        self.rb6.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#1f77b4;
                                                                                 line-height:26px '''
        )
        self.pushButton_start_test = QtWidgets.QPushButton(
            self.widget_model_radio)
        self.pushButton_start_test.move(375, 12)
        self.pushButton_start_test.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_start_test.setMaximumSize(QtCore.QSize(100, 40))
        self.pushButton_start_test.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;font-weight:500;font-size:16px}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_start_test.setText("Start")
        self.pushButton_start_test.setFont(font)

        self.widget_test1 = QtWidgets.QWidget(self.widget_5)
        self.widget_test1.setObjectName("widget_test1")
        self.widget_test1.setGeometry(QtCore.QRect(10, 110, 562, 400))
        self.TESTDisplayGB1 = QtWidgets.QGroupBox(self.widget_test1)
        self.TESTDisplayGB1.setFont(font)
        self.TESTDisplayGB1.setStyleSheet('''border:1px solid #FFFFFF ''')
        self.TESTDisplayGB1.setAlignment(QtCore.Qt.AlignCenter)
        self.fig_test1 = plt.Figure(figsize=(5.0, 2))
        self.test1_canva = FigureCanvas(self.fig_test1)
        self.fig_test1.subplots_adjust(top=0.95,
                                       bottom=0.155,
                                       left=0.09,
                                       right=0.978,
                                       hspace=0.2,
                                       wspace=0.2)
        self.axes_test1 = self.test1_canva.figure.subplots()
        self.test_before = QtWidgets.QLabel()
        self.test_before.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#333333;
                                                                                line-height:26px '''
        )
        self.test_before.setText("Before Data Augmentation")
        self.test_visual1 = QtWidgets.QVBoxLayout()
        self.test_visual1.setContentsMargins(0, 0, 0, 0)
        self.evaluate_before = QtWidgets.QLabel(self.widget_5)
        self.evaluate_before.setGeometry(QtCore.QRect(205, 335, 350, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.evaluate_before.sizePolicy().hasHeightForWidth())
        self.evaluate_before.setSizePolicy(sizePolicy)
        self.evaluate_before.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#333333;
                                                                                        line-height:26px '''
        )
        self.evaluate_before.setText("Evaluate Index：")
        self.test_visual1.addWidget(self.test_before)
        self.test_visual1.addWidget(self.test1_canva)
        self.TESTDisplayGB1.setLayout(self.test_visual1)
        self.widget_test2 = QtWidgets.QWidget(self.widget_5)
        self.widget_test2.setObjectName("widget_test2")
        self.widget_test2.setGeometry(QtCore.QRect(10, 368, 562, 440))
        self.TESTDisplayGB2 = QtWidgets.QGroupBox(self.widget_test2)
        self.TESTDisplayGB2.setFont(font)
        self.TESTDisplayGB2.setStyleSheet('''border:1px solid #FFFFFF ''')
        self.TESTDisplayGB2.setAlignment(QtCore.Qt.AlignCenter)
        self.fig_test2 = plt.Figure(figsize=(5, 2))
        self.test2_canva = FigureCanvas(self.fig_test2)
        self.fig_test2.subplots_adjust(top=0.95,
                                       bottom=0.155,
                                       left=0.09,
                                       right=0.978,
                                       hspace=0.2,
                                       wspace=0.2)
        self.axes_test2 = self.test2_canva.figure.subplots()
        self.test_after = QtWidgets.QLabel()
        self.test_after.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:16px;font-weight:400;color:#333333;
                                                                                        line-height:26px '''
        )
        self.test_after.setText("After Data Augmentation")
        self.test_visual2 = QtWidgets.QVBoxLayout()
        self.test_visual2.setContentsMargins(0, 0, 0, 0)
        self.evaluate_after = QtWidgets.QLabel(self.widget_5)
        self.evaluate_after.setGeometry(QtCore.QRect(205, 598, 350, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(
            self.evaluate_after.sizePolicy().hasHeightForWidth())
        self.evaluate_after.setSizePolicy(sizePolicy)
        self.evaluate_after.setStyleSheet(
            '''font-family:"Microsoft YaHei UI";font-size:14px;font-weight:400;color:#333333;
                                                                                                line-height:26px '''
        )
        self.evaluate_after.setText("Evaluate Index：")
        self.test_visual2.addWidget(self.test_after)
        self.test_visual2.addWidget(self.test2_canva)
        self.pushButton_saveresult = QtWidgets.QPushButton(self.widget_5)
        self.pushButton_saveresult.setGeometry(QtCore.QRect(165, 640, 200, 40))
        self.pushButton_saveresult.setStyleSheet(
            '''QPushButton{background:#52626C;border-radius:6px;color:#ffffff;font-weight:500;font-size:16px}QPushButton:hover{background:#7effb1;} '''
        )
        self.pushButton_saveresult.setText("Save Test Result")
        self.pushButton_saveresult.setFont(font)
        self.TESTDisplayGB2.setLayout(self.test_visual2)
        self.horizontalLayout_2.addWidget(self.widget_5)
        self.verticalLayout.addWidget(self.widget_2)

        self.pushButton_GNI.clicked.connect(Form.click_GNI)
        self.pushButton_MSI.clicked.connect(Form.click_MSI)
        self.pushButton_SNI.clicked.connect(Form.click_SNI)
        self.pushButton_PNI.clicked.connect(Form.click_PNI)
        self.pushButton_COUTOUT.clicked.connect(Form.click_CUTOUT)
        self.pushButton_SMOTE.clicked.connect(Form.click_SMOTE)
        self.pushButton_KMEANS_SMOTE.clicked.connect(Form.click_KMEANS_SMOTE)
        self.pushButton_LLE.clicked.connect(Form.click_LLE)
        self.pushButton_MIXUP.clicked.connect(Form.click_MIXUP)
        self.pushButton_MTD.clicked.connect(Form.click_MTD)
        self.pushButton_KNNMTD.clicked.connect(Form.click_KNNMTD)
        self.pushButton_GMM.clicked.connect(Form.click_GMM)
        self.pushButton_GAN.clicked.connect(Form.click_GAN)
        self.pushButton_WGAN_GP.clicked.connect(Form.click_WGAN_GP)
        self.pushButton_LSGAN.clicked.connect(Form.click_LSGAN)
        self.pushButton_VAE.clicked.connect(Form.click_VAE)
        self.pushButton_VAEGAN.clicked.connect(Form.click_VAEGAN)
        self.pushButton_DDPM.clicked.connect(Form.click_DDPM)
        self.pushButton_MAF.clicked.connect(Form.click_MAF)
        self.pushButton_REALNVP.clicked.connect(Form.click_REALNVP)
        self.pushButton_GLOW.clicked.connect(Form.click_GLOW)
        self.rb1.toggled.connect(Form.select_test_model)
        self.rb2.toggled.connect(Form.select_test_model)
        self.rb3.toggled.connect(Form.select_test_model)
        self.rb4.toggled.connect(Form.select_test_model)
        self.rb5.toggled.connect(Form.select_test_model)
        self.rb6.toggled.connect(Form.select_test_model)
        self.pushButton_start.clicked.connect(Form.begin_gen)
        self.pushButton_start_test.clicked.connect(Form.simulate)
        self.pushButton_savedata.clicked.connect(Form.save_data)
        self.pushButton_savemodel.clicked.connect(Form.save_model)
        self.pushButton_saveresult.clicked.connect(Form.save_test_result)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Logo"))
        self.label_2.setText(_translate("Form", "TextLabel"))
        self.pushButton.setToolTip(
            _translate("Form", "<html><head/><body><p>最小化</p></body></html>"))
        self.pushButton.setText(_translate("Form", "-"))
        self.pushButton_3.setToolTip(
            _translate("Form", "<html><head/><body><p>关闭</p></body></html>"))
        self.pushButton_3.setText(_translate("Form", "X"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.scroll1),
                                 _translate("MainWindow", "Transformation"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.scroll2),
                                 _translate("MainWindow", "Interpolation"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.scroll3),
                                 _translate("MainWindow", "Distribution Estimation"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_T7),
                                 _translate("MainWindow", "Back"))
