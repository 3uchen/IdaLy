#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：3uchen time:2022/12/5
import sys
import time
import torch
import os
import numpy as np
from sklearn.metrics import r2_score
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot, Qt, QPoint
from PyQt5.QtGui import QEnterEvent, QPixmap, QIcon, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
import matplotlib.pyplot as plt
from ui_idap_v1 import Ui_Form, Ui_Gen_Visual
import test
import execute
from res import description


method_set = {'GNI': execute.GNI_execute, 'MSI': execute.MSI_execute,
              'SNI': execute.SNI_execute, 'PNI': execute.PNI_execute,
              'CUTOUT': execute.Cutout_execute,
              'SMOTE': execute.SMOTE_execute, 'LLE': execute.LLE_execute,
              'KMEANS_SMOTE': execute.Kmeans_SMOTE_execute, 'MIXUP': execute.MIXUP_execute,
              'KNNMTD': execute.KNNMTD_execute, 'MTD': execute.MTD_execute,
              'GMM': execute.GMM_execute, 'GAN': execute.GAN_execute,
              'VAE': execute.VAE_execute, 'WGAN_GP': execute.WGAN_GP_execute,
              'DDPM': execute.DDPM_execute, 'LSGAN': execute.LSGAN_execute, 
              'VAEGAN': execute.VAEGAN_execute, 'MAF': execute.MAF_execute,
              'REALNVP': execute.REALNVP_execute, 'GLOW': execute.GLOW_execute}
model_need_dic = {'GNI': False, 'MSI': False, 'SNI': False, 'PNI': False, 'CUTOUT':False,
                  'SMOTE': False, 'KMEANS_SMOTE': False, 'MIXUP': False,
                  'LLE': False, 'KNNMTD': False, 'MTD': False, 'GMM': False,
                  'GAN': True, 'VAE': True, 'VAEGAN': True,
                  'WGAN_GP': True, 'DDPM': True, 'LSGAN': True,
                  'MAF': True, 'REALNVP': True, 'GLOW': True}
model_name = {'GAN': ['netG', 'netD'], 'WGAN_GP': ['netG', 'netD'],
              'VAE': ['VAE'], 'DDPM': ['DDPM'], 'LSGAN': ['netG', 'netD'], 'VAEGAN': ['net_encoder', 'net_decoder', 'net_discriminator'],
              'MAF': ['MAF'], 'REALNVP': ['REALNVP'], 'GLOW': ['GLOW']}
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
         '#e377c2', '#7f7f7f',
         '#bcbd22', '#17becf']


class SVR_thread(QtCore.QThread):
    sinout = QtCore.pyqtSignal(list, list)

    def __init__(self, svr_ori, svr_new):
        super().__init__()
        self.svr_ori = svr_ori
        self.svr_new = svr_new

    def run(self):
        pre_ori, pre_new = test.svr_simulate(original_data=self.svr_ori,
                                             new_data=self.svr_new)
        pre_ori = pre_ori.tolist()
        pre_new = pre_new.tolist()
        self.sinout.emit(pre_ori, pre_new)


class Simulate_Thread(QtCore.QThread):
    sinout_simulate = QtCore.pyqtSignal(list, list)

    def __init__(self, simulate_model, ori_data, new_data, test_data,
                 data_type):
        super().__init__()
        self.ori_data = ori_data
        self.new_data = new_data
        self.test_data = test_data
        self.simulate_model = simulate_model
        self.data_type = data_type

    def run(self):
        test_x = self.test_data[:, :-1]
        if self.data_type == 1:
            pre_ori, pre_new = test.fault_diagnosis_simulate(
                simulate_model=self.simulate_model,
                original_data=self.ori_data,
                new_data=self.new_data, test_x=test_x)
        if self.data_type == 0:
            pre_ori, pre_new = test.soft_sensor_simulate(
                simulate_model=self.simulate_model,
                original_data=self.ori_data,
                new_data=self.new_data, test_x=test_x)
        pre_ori = pre_ori.tolist()
        pre_new = pre_new.tolist()
        self.sinout_simulate.emit(pre_ori, pre_new)


class Gen_thread(QtCore.QThread):
    sinout_gen = QtCore.pyqtSignal(list, list)

    def __init__(self, original_data, num_gen, para, method_name, data_type, label_need):
        super(Gen_thread, self).__init__()
        self.model_need = model_need_dic[method_name]
        self.original_data = original_data
        self.num_gen = num_gen
        self.para = para
        self.fault_num = len(num_gen)
        self.method_name = method_name
        self.data_type = data_type
        self.label_need = label_need

    def run(self):
        gen_data, model = [], []
        if self.data_type == 0:
            if self.model_need:
                model_s, gen_data_s = method_set[self.method_name](
                    original_data=self.original_data[0],
                    num_gen=self.num_gen[0],
                    para=self.para)
                model.append(model_s)
                gen_data.append(gen_data_s)
            else:
                gen_data.append(
                    method_set[self.method_name](original_data=self.original_data[0],
                                                 num_gen=self.num_gen[0],
                                                 para=self.para))
        elif self.data_type == 1:
            if self.label_need:
                if self.model_need:
                    data_dim = self.original_data[0].shape[1]
                    original_data = np.array(self.original_data)
                    original_data = np.reshape(original_data, (-1, data_dim))
                    model, data_gen = method_set[self.method_name](original_data=original_data,
                                                                   num_gen=self.num_gen,
                                                                   para=self.para)
                else:
                    data_gen = method_set[self.method_name](original_data=self.original_data,
                                                            num_gen=self.num_gen,
                                                            para=self.para)
                gen_data = data_gen
            else:
                for i in range(self.fault_num):
                    if self.num_gen[i] <= 0:
                        continue
                    y_gen = np.full((self.num_gen[i], 1), i + 1, dtype=int)
                    if self.model_need:
                        model_i, x_gen = method_set[self.method_name](original_data=self.original_data[i][:, :-1],
                                                                    num_gen=self.num_gen[i],
                                                                    para=self.para)
                    else:
                        x_gen = method_set[self.method_name](original_data=self.original_data[i][:, :-1],
                                                            num_gen=self.num_gen[i],
                                                            para=self.para)
                    gen_i = np.concatenate([x_gen, y_gen], axis=1)
                    gen_data.append(gen_i)
                    if self.model_need:
                        model.append(model_i)
        self.sinout_gen.emit(model, gen_data)


class Gen_visual(QWidget, Ui_Gen_Visual):
    def __init__(self, parent=None) -> None:
        super(Gen_visual, self).__init__(parent)
        self.setupui(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
    

class IDAP(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(IDAP, self).__init__(parent)
        self.setupUi(self)
        self._init_main_window()
        self.stackedWidget.setCurrentIndex(0)
        self._initDrag()
        self.setMouseTracking(True)
        self._close_max_min_icon()
        self.my_Qss()
        self.widget.installEventFilter(self)
        self.widget_2.installEventFilter(self)
        self.original_data = None
        self.original_train = []
        self.original_train_array = None
        self.test_data = None
        self.gen_data = None
        self.data_gen_list = None
        self.all_train = None
        self.method_index = None
        self.method_name = None
        self.para_dic = {'GNI': [self.mean_value, self.var_value],
                         'MSI': [self.mp_value_msi],
                         'SNI': [self.mean_value, self.var_value],
                         'PNI': [],
                         'CUTOUT': [self.mp_value_cutout],
                         'SMOTE': [self.kn_value_smote],
                         'KMEANS_SMOTE': [self.num_clusters_value_kmeans_smote],
                         'MIXUP': [self.alpha_value],
                         'LLE': [self.kn_value_lle, self.reg_value,
                                 self.con_value],
                         'KNNMTD': [self.kn_value_knnmtd],
                         'MTD': [],
                         'GMM': [self.con_value_gmm],
                         'GAN': [self.epoch_value_gan_vae,
                                 self.lr_value_gan_vae,
                                 self.batch_value_gan_vae,
                                 self.latent_value_gan_vae],
                         'WGAN_GP': [self.iter_value_wgan_gp,
                                     self.lr_value_wgan_gp,
                                     self.batch_value_wgan_gp,
                                     self.latent_value_wgan_gp,
                                     self.n_critic_value_wgan_gp,
                                     self.lambda_value_wgan_gp],
                         'LSGAN': [self.iter_value_lsgan,
                                   self.lr_value_lsgan,
                                   self.batch_value_lsgan,
                                   self.latent_value_lsgan,
                                   self.n_critic_value_lsgan],
                         'VAE': [self.epoch_value_gan_vae,
                                 self.lr_value_gan_vae,
                                 self.batch_value_gan_vae,
                                 self.latent_value_gan_vae],
                         'VAEGAN': [self.epoch_value_gan_vae,
                                    self.lr_value_gan_vae,
                                    self.batch_value_gan_vae,
                                    self.latent_value_gan_vae],
                         'DDPM': [self.epoch_value_ddpm,
                                  self.lr_value_ddpm,
                                  self.batch_value_ddpm,
                                  self.num_steps_value_ddpm],
                         'MAF':  [self.epoch_value_flow,
                                  self.lr_value_flow,
                                  self.batch_value_flow,
                                  self.num_blocks_value_flow],
                         'REALNVP':  [self.epoch_value_flow,
                                      self.lr_value_flow,
                                      self.batch_value_flow,
                                      self.num_blocks_value_flow],
                         'GLOW':  [self.epoch_value_flow,
                                   self.lr_value_flow,
                                   self.batch_value_flow,
                                   self.num_blocks_value_flow],}
        self.para_set = []
        self.Model_gen = None
        self.Model_svr_ori = None
        self.Model_svr_new = None
        self.data_type = 0
        self.fault_num = 0
        self.test_num = 100000000
        self.gen_num = []
        self.ori_num = []
        self.pre_ori = []
        self.pre_new = []
        self.fault_real_train = 0
        self.gen_flag = False
        self.match_flag = True
        self.description = description
        self.simulate_model = None
        self.label_need = False

    def click_GNI(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.setPlainText(self.description[0] + self.description[1])
        self.stackedWidget.setCurrentIndex(1)
        self.method_name = 'GNI'
        return
    
    def click_MSI(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.setPlainText(self.description[2] + self.description[3])
        self.stackedWidget.setCurrentIndex(2)
        self.method_name = 'MSI'
        return

    def click_SNI(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.setPlainText(self.description[4] + self.description[5])
        self.stackedWidget.setCurrentIndex(1)
        self.method_name = 'SNI'
        return
    
    def click_PNI(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.setPlainText(self.description[6] + self.description[7])
        self.stackedWidget.setCurrentIndex(0)
        self.method_name = 'PNI'
        return
    
    def click_CUTOUT(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.setPlainText(self.description[40] + self.description[41])
        self.stackedWidget.setCurrentIndex(3)
        self.method_name = 'CUTOUT'
        return
    
    def click_SMOTE(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[8] + self.description[9])
        self.stackedWidget.setCurrentIndex(4)
        self.method_name = 'SMOTE'
        return
    
    def click_KMEANS_SMOTE(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[10] + self.description[11])
        self.stackedWidget.setCurrentIndex(5)
        self.method_name = 'KMEANS_SMOTE'
        return

    def click_LLE(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[12] + self.description[13])
        self.stackedWidget.setCurrentIndex(6)
        self.method_name = 'LLE'
        return
    
    def click_MIXUP(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[14] + self.description[15])
        self.stackedWidget.setCurrentIndex(7)
        self.method_name = 'MIXUP'
        return

    def click_KNNMTD(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[16] + self.description[17])
        self.stackedWidget.setCurrentIndex(8)
        self.method_name = 'KNNMTD'
        return

    def click_MTD(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[18] + self.description[19])
        self.stackedWidget.setCurrentIndex(0)
        self.method_name = 'MTD'

        return

    def click_GMM(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[20] + self.description[21])
        self.stackedWidget.setCurrentIndex(9)
        self.method_name = 'GMM'
        return

    def click_GAN(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[22] + self.description[23])
        self.stackedWidget.setCurrentIndex(10)
        self.method_name = 'GAN'
        return
      
    def click_WGAN_GP(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[24] + self.description[25])
        self.stackedWidget.setCurrentIndex(11)
        self.method_name = 'WGAN_GP'
        return

    def click_VAE(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[26] + self.description[27])
        self.stackedWidget.setCurrentIndex(10)
        self.method_name = 'VAE'
        return
    
    def click_LSGAN(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[28] + self.description[29])
        self.stackedWidget.setCurrentIndex(12)
        self.method_name = 'LSGAN'
        return
    
    def click_VAEGAN(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[30] + self.description[31])
        self.stackedWidget.setCurrentIndex(10)
        self.method_name = 'VAEGAN'
        return
    
    def click_MAF(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[32] + self.description[33])
        self.stackedWidget.setCurrentIndex(13)
        self.method_name = 'MAF'
        return
    
    def click_REALNVP(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[34] + self.description[35])
        self.stackedWidget.setCurrentIndex(13)
        self.method_name = 'REALNVP'
        return
    
    def click_GLOW(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[36] + self.description[37])
        self.stackedWidget.setCurrentIndex(13)
        self.method_name = 'GLOW'
        return
    
    def click_DDPM(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.insertHtml(self.description[38] + self.description[39])
        self.stackedWidget.setCurrentIndex(14)
        self.method_name = 'DDPM'
        return

    def trigger_actHelp(self): 
        QMessageBox.about(self, "About",
                          """Industrial Data Augmentation System v1.0""")
        return

    def begin_gen(self):
        if self.original_data is None:
            box1 = QMessageBox()
            box1.about(self, "Remind",
                       """Data not imported！""")
            return
        if self.num_gen_value.value() == 0:
            box1 = QMessageBox()
            box1.about(self, "Remind",
                       """The number of generated samples can't be 0！""")
            return
        if self.method_name == None:
            box1 = QMessageBox()
            box1.about(self, "Remind",
                       """Method not selected！""")
            return
        self.gen_flag = True
        for para in self.para_dic[self.method_name]:
            self.para_set.append(para.value())
        for ori_data in self.original_train:
            self.ori_num.append(len(ori_data))
        self.fault_num = len(self.ori_num)
        min_num_ori = min(self.ori_num)
        balance_num = min_num_ori + self.num_gen_value.value()
        for num_i in self.ori_num:
            if balance_num - num_i <= 0:
                self.gen_num.append(0)
            else:
                self.gen_num.append(balance_num - num_i)
        self.pushButton_start.setEnabled(False)
        self.gen_thread = Gen_thread(original_data=self.original_train, num_gen=self.gen_num, para=self.para_set,
                                     method_name=self.method_name, data_type=self.data_type, label_need=self.label_need)
        self.gen_thread.sinout_gen.connect(self.gen_data_get)
        self.gen_thread.start()
        # self.gen_thread.join()

    def gen_data_get(self, model_list, data_gen_list):
        self.Model_gen = model_list
        self.data_gen_list = data_gen_list
        if self.data_type == 0:
            self.gen_data = data_gen_list[0]
            self.original_train_array = self.original_train[0]
        elif self.data_type == 1:
            self.gen_data = data_gen_list[0]
            self.original_train_array = self.original_train[0]
            for i in range(1, self.fault_num):
                self.gen_data = np.concatenate([self.gen_data, data_gen_list[i]], axis=0)
                self.original_train_array = np.concatenate([self.original_train_array, self.original_train[i]], axis=0)

        self.all_train = np.concatenate((self.original_train_array, self.gen_data), axis=0)
        self.visualization()
        self.match_flag = False
        self.para_set.clear()
        self.ori_num.clear()
        self.gen_num.clear()
        box2 = QMessageBox()
        box2.about(self, "Remind",
                   """Data generation done！""")
        self.gen_thread.quit()
        self.pushButton_start.setEnabled(True)

    def select_test_model(self):
        if self.rb1.isChecked():
            self.simulate_model = 'SVM'
        if self.rb2.isChecked():
            self.simulate_model = 'KNN'
        if self.rb3.isChecked():
            self.simulate_model = 'MLP'
        if self.rb4.isChecked():
            self.simulate_model = 'DecisionTree'
        if self.rb5.isChecked():
            self.simulate_model = 'RadomForest'
        if self.rb6.isChecked():
            self.simulate_model = 'AdaBoost'
        print(self.simulate_model)

    def simulate(self, ):
        if self.simulate_model is None:
            box2 = QMessageBox()
            box2.about(self, "Remind",
                       """Test model hasn't been selected!""")
            return
        if not self.gen_flag:
            box2 = QMessageBox()
            box2.about(self, "Remind",
                       """Data augmentation hasn't been performed！""")
            return
        self.pushButton_start_test.setEnabled(False)
        np.random.shuffle(self.original_train_array)
        np.random.shuffle(self.all_train)
        self.simulate_thread = Simulate_Thread(
            simulate_model=self.simulate_model,
            ori_data=self.original_train_array,
            new_data=self.all_train,
            test_data=self.test_data,
            data_type=self.data_type)
        self.simulate_thread.sinout_simulate.connect(self.simulate_visual)
        self.simulate_thread.start()

    def simulate_visual(self, pre_ori, pre_new):
        self.match_flag = True
        test_y = self.test_data[:, -1]
        self.pre_ori = pre_ori
        self.pre_new = pre_new
        x = list(range(len(test_y)))
        self.axes_test1.cla()
        self.axes_test2.cla()
        if self.data_type == 0:
            r2_ori = round(r2_score(test_y, pre_ori), 4)
            r2_new = round(r2_score(test_y, pre_new), 4)
            self.axes_test1.plot(x, test_y, c=color[0], label='y_true')
            self.axes_test1.plot(x, pre_ori, c=color[1], label='y_pred')
            self.axes_test2.plot(x, test_y, c=color[0], label='y_true')
            self.axes_test2.plot(x, pre_new, c=color[1], label='y_pred')
            self.axes_test1.legend(frameon=True)
            self.axes_test2.legend(frameon=True)
            self.test1_canva.draw()
            self.test2_canva.draw()
            self.evaluate_before.setText('R2_Score:  ' + str(r2_ori))
            self.evaluate_after.setText('R2_Score:  ' + str(r2_new))
            self.simulate_thread.quit()
        elif self.data_type == 1:
            sum_ori, sum_new = 0, 0
            for i in range(len(test_y)):
                if test_y[i] == pre_ori[i]:
                    sum_ori = sum_ori + 1
                if test_y[i] == pre_new[i]:
                    sum_new = sum_new + 1
            acc_ori = round((sum_ori / len(test_y)) * 100, 4)
            acc_new = round((sum_new / len(test_y)) * 100, 4)
            y = list(range(1, self.fault_num + 1, 1))
            x_tick = list(range(0, (self.fault_num + 1) * self.test_num, self.test_num))
            y_tick = ['class{}'.format(i + 1) for i in range(self.fault_num)]
            self.axes_test1.scatter(x, test_y, c='black', marker='o', alpha=0.8, s=55, label='y_true')
            self.axes_test1.scatter(x, pre_ori, c='red', marker='o', alpha=0.5, s=10, label='y_pred')
            self.axes_test2.scatter(x, test_y, c='black', marker='o', alpha=0.8, s=55, label='y_true')
            self.axes_test2.scatter(x, pre_new, c='red', marker='o', alpha=0.5, s=10, label='y_pred')
            self.axes_test1.set_yticks(y)
            self.axes_test1.set_xticks(x_tick)
            self.axes_test1.set_yticklabels(y_tick, fontsize=7)
            self.axes_test2.set_yticks(y)
            self.axes_test2.set_xticks(x_tick)
            self.axes_test2.set_yticklabels(y_tick, fontsize=7)
            self.axes_test1.grid(True, linestyle="--", color="gray", linewidth="0.5", axis="both")
            self.axes_test2.grid(True, linestyle="--", color="gray", linewidth="0.5", axis="both")
            self.test1_canva.draw()
            self.test2_canva.draw()
            self.evaluate_before.setText('Accuracy: ' + str(acc_ori))
            self.evaluate_after.setText('Accuracy: ' + str(acc_new))
            self.simulate_thread.quit()
        self.pushButton_start_test.setEnabled(True)

    def openfile(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self,
                                                         "getOpenFileName", "C:/",
                                                         "All Files (*);;Text Files (*.txt)")
        if filename[0] == '':
            return
        self.original_data = np.load(filename[0])
        data = self.original_data
        y_ori = self.original_data[:, -1]
        box2 = QMessageBox()
        self.original_train = []
        if y_ori[0] == int(y_ori[0]):
            self.data_type = 1
            box2.about(self, "Remind",
                       """You have selected data used for fault diagnosis""")
            data[:, :-1] = (data[:, :-1] - data[:, :-1].min(0)) / (data[:, :-1].max(0) - data[:, :-1].min(0))
            fault_num = max(data[:, -1])
            data_dim = data.shape[1]
            test_data = []
            self.test_num = 100
            for i in range(int(fault_num)):
                data_i = data[data[:, -1] == i + 1]
                np.random.shuffle(data_i)
                if data_i[:, -1][0] == 1:
                    data_i[:, -1] = 1
                    test_data.append(data_i[:100])
                    self.original_train.append(data_i[100:150])
                elif data_i[:, -1][0] == 3:
                    data_i[:, -1] = 2
                    test_data.append(data_i[:100])
                    self.original_train.append(data_i[100:350]) 
                elif data_i[:, -1][0] == 4:
                    data_i[:, -1] = 3
                    test_data.append(data_i[:100])
                    self.original_train.append(data_i[100:600])
                elif data_i[:, -1][0] == 5:
                    data_i[:, -1] = 4
                    test_data.append(data_i[:100])
                    self.original_train.append(data_i[100:250])         
            test_data = np.array(test_data)
            self.test_data = np.reshape(test_data, (-1, data_dim))
            self.fault_num = len(self.original_train)
        else:
            self.data_type = 0
            box2.about(self, "Remind",
                       """You have selected data used for soft sensor""")
            np.random.shuffle(data)
            self.original_train.append(data[:int(0.7 * len(data)), :])
            self.test_data = data[int(0.7 * len(data)):, :]

    def save_data_pro(self, directoryroute):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if self.data_type == 0:
            data_gen = self.data_gen_list[0]
            filename = self.method_name + '_soft_sensor_' + 'data_gen' + '_' + timestr
            filepath = os.path.join(directoryroute, filename)
            np.save(filepath, data_gen)
        if self.data_type == 1:
            for i in range(len(self.data_gen_list)):
                data_class_i = self.data_gen_list[i]
                filename = self.method_name + '_fault_diagnosis_' + 'data_gen' + \
                           '_class{}_'.format(i + 1) + timestr
                filepath = os.path.join(directoryroute, filename)
                np.save(filepath, data_class_i)

    def save_data(self):
        if not self.gen_flag:
            box2 = QMessageBox()
            box2.about(self, "Remind",
                       """Data augmentation hasn't been performed！""")
            return
        directoryroute = QtWidgets.QFileDialog.getExistingDirectory(None, "Choose path of file saved", "C:\\")
        self.save_data_pro(directoryroute)
        box2 = QMessageBox()
        box2.about(self, "Remind",
                   """The data generated has been saved successfully！""")

    def save_model_pro(self, directoryroute):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if self.data_type == 0:
            model_ = self.Model_gen[0]
            for i in range(len(model_)):
                model_i = model_[i]
                filename = self.method_name + '_soft_sensor_' + model_name[self.method_name][i] + '_' + timestr + '.pkl'
                filepath = os.path.join(directoryroute, filename)
                torch.save(model_i.state_dict(), filepath)
        if self.data_type == 1:
            for i in range(len(self.Model_gen)):
                model_class_i = self.Model_gen[i]
                for j in range(len(model_class_i)):
                    model_j = model_class_i[j]
                    filename = self.method_name + '_fault_diagnosis_' + model_name[self.method_name][j]+ \
                                '_class{}_'.format(i + 1) + timestr + '.pkl '
                    filepath = os.path.join(directoryroute, filename)
                    torch.save(model_j.state_dict(), filepath)

    def save_model(self):
        if not self.gen_flag:
            box2 = QMessageBox()
            box2.about(self, "Remind",
                       """Data augmentation hasn't been performed！""")
            return
        if not self.Model_gen:
            box2 = QMessageBox()
            box2.about(self, "Remind",
                       """The method you choose doesn't need to use model！""")
            return
        directoryroute = QtWidgets.QFileDialog.getExistingDirectory(None, "Choose path of file saved", "C:\\")
        self.save_model_pro(directoryroute)
        box2 = QMessageBox()
        box2.about(self, "Remind",
                   """The generative model has been saved successfully！""")

    def save_test_result_pro(self, directoryroute):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if self.data_type == 0:
            plt.figure(figsize=(15, 10))
            test_y = self.test_data[:, -1]
            x = list(range(len(test_y)))
            ax1 = plt.subplot(211)
            ax1.plot(x, test_y, c=color[0], label='y_true')
            ax1.plot(x, self.pre_ori, c=color[1], label='y_pred')
            ax1.legend(frameon=True, fontsize=12)
            ax1.set_title("Before Data Augmentation(R2_Score: {})".format(round(r2_score(test_y, self.pre_ori), 4)),
                          fontdict={"family": "Microsoft YaHei UI", "size": 18}, pad=10)
            ax2 = plt.subplot(212)
            ax2.plot(x, test_y, c=color[0], label='y_true')
            ax2.plot(x, self.pre_new, c=color[1], label='y_pred')
            ax2.legend(frameon=True, fontsize=12)
            ax2.set_xlabel("Sample number", fontsize=15)
            ax1.set_ylabel("Value of key variable", fontsize=15)
            ax2.set_ylabel("Value of key variable", fontsize=15)
            ax2.set_title("After Data Augmentation(R2_Score: {})".format(round(r2_score(test_y, self.pre_new), 4)),
                          fontdict={"family": "Microsoft YaHei UI", "size": 18}, pad=10)
            filename = self.method_name + '_'+self.simulate_model+'_soft_sensor_test_result_' + timestr + '.png'
            filepath = os.path.join(directoryroute, filename)
            plt.savefig(filepath)
        if self.data_type == 1:
            sum_ori, sum_new = 0, 0
            test_y = self.test_data[:, -1]
            for i in range(len(test_y)):
                if test_y[i] == self.pre_ori[i]:
                    sum_ori = sum_ori + 1
                if test_y[i] == self.pre_new[i]:
                    sum_new = sum_new + 1
            acc_ori = round((sum_ori / len(test_y)) * 100, 4)
            acc_new = round((sum_new / len(test_y)) * 100, 4)
            plt.figure(figsize=(15, 10))
            x = list(range(len(test_y)))
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            y = list(range(1, self.fault_num + 1, 1))
            x_tick = list(range(0, (self.fault_num + 1) * self.test_num, self.test_num))
            y_tick = ['class{}'.format(i + 1) for i in range(self.fault_num)]
            ax1.scatter(x, test_y, c='black', marker='o', alpha=0.8, s=55, label='y_true')
            ax1.scatter(x, self.pre_ori, c='red', marker='o', alpha=0.5, s=10, label='y_pred')
            ax2.scatter(x, test_y, c='black', marker='o', alpha=0.8, s=55, label='y_true')
            ax2.scatter(x, self.pre_new, c='red', marker='o', alpha=0.5, s=10, label='y_pred')
            ax1.set_yticks(y)
            ax1.set_xticks(x_tick)
            ax1.set_yticklabels(y_tick, fontsize=13)
            ax2.set_yticks(y)
            ax2.set_xticks(x_tick)
            ax2.set_yticklabels(y_tick, fontsize=13)
            ax2.set_xlabel("Samples", fontsize=15)
            ax1.grid(True, linestyle="--", color="gray", linewidth="0.5", axis="both")
            ax2.grid(True, linestyle="--", color="gray", linewidth="0.5", axis="both")
            ax1.set_title("Before Data Augmentation(Accuracy: {})".format(acc_ori),
                          fontdict={"size": 18}, pad=10)
            ax2.set_title("After Data Augmentation(Accuracy: {})".format(acc_new),
                          fontdict={"size": 18}, pad=10)
            filename = self.method_name + '_' + self.simulate_model + '_fault_diagnosis_test_result_' + timestr + '.png'
            filepath = os.path.join(directoryroute, filename)
            plt.savefig(filepath)

    def save_test_result(self):
        if not self.gen_flag:
            box2 = QMessageBox()
            box2.about(self, "Remind",
                       """Data augmentation hasn't been performed！""")
            return
        directoryroute = QtWidgets.QFileDialog.getExistingDirectory(None, "Choose path of file saved", "C:\\")
        self.save_test_result_pro(directoryroute)
        box2 = QMessageBox()
        box2.about(self, "Remind",
                   """The test result has been saved successfully！""")
        return

    def save_all(self):
        if not self.gen_flag:
            box2 = QMessageBox()
            box2.about(self, "Remind",
                       """Data augmentation hasn't been performed！""")
            return
        if not self.match_flag:
            box2 = QMessageBox()
            box2.about(self, "Remind",
                       """The results bewteen data augmentation and model test don't match each other！""")
            return
        timestr = time.strftime("%Y%m%d-%H%M%S")
        directoryroute = QtWidgets.QFileDialog.getExistingDirectory(None, "Choose path of file saved", "C:\\")
        print(directoryroute)
        os.makedirs(directoryroute + '\Data_augmentation_saved' + timestr + '\Data_generated')
        self.save_data_pro(directoryroute + '\Data_augmentation_saved' + timestr + '\Data_generated')
        if self.Model_gen:
            os.makedirs(directoryroute + '\Data_augmentation_saved' + timestr + '\Model_trained')
            self.save_model_pro(directoryroute + '\Data_augmentation_saved' + timestr + '\Model_trained')
        os.makedirs(directoryroute + '\Data_augmentation_saved' + timestr + '\Test_result')
        self.save_test_result_pro(directoryroute + '\Data_augmentation_saved' + timestr + '\Test_Result')
        box2 = QMessageBox()
        box2.about(self, "Remind",
                   """Saved successfully！""")

    def visualization(self):
        from sklearn.decomposition import PCA
        p = PCA(n_components=2)
        if self.data_type == 1:
            all_train_x = self.all_train[:, :-1]
            pca = p.fit_transform(all_train_x)
        elif self.data_type == 0:
            pca = p.fit_transform(self.all_train)
        pca_data_gen = pca[sum(self.ori_num):]
        i_sum_real, i_sum_gen = 0, 0
        self.axes_pca.cla()
        if self.data_type == 1:
            scatter_gen, scatter_ori = [], []
            for i in range(len(self.ori_num)):
                real_i = pca[i_sum_real:i_sum_real + self.ori_num[i]]
                gen_i = pca_data_gen[i_sum_gen:i_sum_gen + self.gen_num[i]]
                np.random.shuffle(real_i)
                np.random.shuffle(gen_i)
                real_i_x = real_i[:, 0]
                real_i_y = real_i[:, 1]
                gen_i_x = gen_i[:, 0]
                gen_i_y = gen_i[:, 1]
                i_sum_gen = i_sum_gen + self.gen_num[i]
                i_sum_real = i_sum_real + self.ori_num[i]
                scatter_gen_i = self.axes_pca.scatter(gen_i_x[:50], gen_i_y[:50], c=color[i],
                                                      marker='o', s=90, alpha=0.25,
                                                      label='gen_{}'.format(i + 1))
                scatter_ori_i = self.axes_pca.scatter(real_i_x[:50], real_i_y[:50], c=color[i],
                                                      marker='o', s=40, alpha=1,
                                                      edgecolors='black',
                                                      label='real_{}'.format(i + 1))
                scatter_gen.append(scatter_gen_i)
                scatter_ori.append(scatter_ori_i)
            scatter_all = self.axes_pca.scatter([0], [0], color='white', edgecolors='black', s=40, alpha=0.001, label='all')
            leg = self.axes_pca.legend(ncol=self.fault_num+1, fontsize=10, frameon=True,
                                       columnspacing=0.15)
            for j in range(2*self.fault_num+1):
                leg.legendHandles[j].set_picker(True)
            
            def onpick(event):
                legline = event.artist
                label = legline.get_label()
                if label == 'all':
                    for i in range(len(scatter_gen)):
                        scatter_gen[i].set_visible(True)
                        scatter_ori[i].set_visible(True)
                else:
                    for i in range(len(scatter_gen)):
                        scatter_gen[i].set_visible(False)
                        scatter_ori[i].set_visible(False)
                    scatter_ori[int(label[-1])-1].set_visible(True)
                    scatter_gen[int(label[-1])-1].set_visible(True)
                self.pca_canva.draw()
            self.pca_canva.mpl_connect('pick_event', onpick)
        elif self.data_type == 0:
            real = pca[:self.ori_num[0]]
            gen = pca[self.ori_num[0]:]
            np.savetxt('real.txt', real)
            np.savetxt('gen.txt', gen)
            real_x = real[:, 0]
            real_y = real[:, 1]
            gen_x = gen[:, 0]
            gen_y = gen[:, 1]
            self.axes_pca.scatter(gen_x[:200], gen_y[:200], c=color[4], marker='o', s=90, alpha=0.25,
                                  label='gen_data')
            self.axes_pca.scatter(real_x[:200], real_y[:200], c=color[4], marker='o', s=40, alpha=1,
                                  edgecolors='black', label='real_data')
            self.axes_pca.legend(frameon=True)
        self.pca_canva.draw()

    def _init_main_window(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.toolBox.setCurrentIndex(4)
        self.pix = QPixmap(":/logo.png")
        self.label.setPixmap(self.pix)
        self.label.setScaledContents(True)
        self.label_2.setText('Industrial Data Augmentation System V1.0')
        self.label_2.setStyleSheet('''
                                    font-family:"Microsoft YaHei UI";
                                   font-size:26px;
                                   font-weight:bold;
                                   color:#FFFFFF;
                                   line-height:39px
                                   ''')
        self.pix_1 = QPixmap(":/1.png")
        self.label_3.setPixmap(self.pix_1)
        self.label_3.setScaledContents(True)
        self.pix_2 = QPixmap(":/open.png")
        self.label_open.setPixmap(self.pix_2)
        self.label_open.setScaledContents(True)
        font = QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        font.setWeight(440)
        for i in range(5):
            self.toolBox.setItemIcon(i, QIcon(":/tab.png"))
            self.toolBox.setStyleSheet("QToolBox{icon-size:28px;border-radius:5px;}")
            self.toolBox.setFont(font)
        self.pix_3 = QPixmap(":/save.png") 
        self.label_save.setPixmap(self.pix_3)
        self.label_save.setScaledContents(True)
        self.label_save.setStyleSheet("border-image:url(save1.png)")
        self.pix_4 = QPixmap(":/help.png")
        self.label_help.setPixmap(self.pix_4)
        self.label_help.setScaledContents(True)
        self.label_4.setText('Method Select')
        self.label_4.setStyleSheet('''
                                    font-family:"Microsoft YaHei UI";
                                    font-size:18px;
                                    font-weight:550;
                                    color:#333333;
                                    line-height:40px
                                    ''')
        self.label_5.setPixmap(self.pix_1)
        self.label_5.setScaledContents(True)
        self.label_6.setText('Description')
        self.label_6.setStyleSheet('''
                                    font-family:"Microsoft YaHei UI";
                                    font-size:18px;
                                    font-weight:550;
                                    color:#333333;
                                    line-height:40px
                                    ''')
        self.label_7.setPixmap(self.pix_1)
        self.label_7.setScaledContents(True)
        self.label_8.setText('Parameter Setting')
        self.label_8.setStyleSheet('''
                                    font-family:"Microsoft YaHei UI";
                                    font-size:18px;
                                    font-weight:550;
                                    color:#333333;
                                    line-height:40px
                                    ''')
        self.label_9.setPixmap(self.pix_1)
        self.label_9.setScaledContents(True)
        self.label_10.setText('Data Generation')
        self.label_10.setStyleSheet('''
                                    font-family:"Microsoft YaHei UI";
                                    font-size:18px;
                                    font-weight:550;
                                    color:#333333;
                                    line-height:40px
                                    ''')
        self.label_11.setPixmap(self.pix_1)
        self.label_11.setScaledContents(True)
        self.label_12.setText('Model Test')
        self.label_12.setStyleSheet('''
                                    font-family:"Microsoft YaHei UI";
                                    font-size:18px;
                                    font-weight:550;
                                    color:#333333;
                                    line-height:40px
                                    ''')
        self.label_open.mousePressEvent = self.open_link
        self.label_save.mousePressEvent = self.save_link
        self.label_help.mousePressEvent = self.help_link

    def open_link(self, test):
        self.openfile()
        return

    def save_link(self, test):
        self.save_all()
        return

    def help_link(self, test):
        self.trigger_actHelp()
        return

    def _initDrag(self):
        self._move_drag = False
        self._corner_drag = False
        self._bottom_drag = False
        self._right_drag = False

    def _close_max_min_icon(self):
        self.pushButton_3.setText('r')
        self.pushButton.setText('0')

    @pyqtSlot()
    def on_pushButton_clicked(self):
        self.showMinimized()

    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        self.close()

    def eventFilter(self, obj, event):
        if isinstance(event, QEnterEvent):
            self.setCursor(Qt.ArrowCursor)
        return super(IDAP, self).eventFilter(obj, event)  

    def resizeEvent(self, QResizeEvent):
        self._right_rect = [QPoint(x, y) for x in range(self.width() - 5, self.width() + 5)
                            for y in range(self.widget.height() + 20, self.height() - 5)]
        self._bottom_rect = [QPoint(x, y) for x in range(1, self.width() - 5)
                             for y in range(self.height() - 5, self.height() + 1)]
        self._corner_rect = [QPoint(x, y) for x in range(self.width() - 5, self.width() + 1)
                             for y in range(self.height() - 5, self.height() + 1)]

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton) and (event.pos() in self._corner_rect):
            self._corner_drag = True
            event.accept()
        elif (event.button() == Qt.LeftButton) and (event.pos() in self._right_rect):
            self._right_drag = True
            event.accept()
        elif (event.button() == Qt.LeftButton) and (event.pos() in self._bottom_rect):
            self._bottom_drag = True
            event.accept()
        elif (event.button() == Qt.LeftButton) and (event.y() < self.widget.height()):
            self._move_drag = True
            self.move_DragPosition = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, QMouseEvent):
        if QMouseEvent.pos() in self._corner_rect:  
            self.setCursor(Qt.SizeFDiagCursor)
        elif QMouseEvent.pos() in self._bottom_rect:
            self.setCursor(Qt.SizeVerCursor)
        elif QMouseEvent.pos() in self._right_rect:
            self.setCursor(Qt.SizeHorCursor)
        if Qt.LeftButton and self._right_drag:
            self.resize(QMouseEvent.pos().x(), self.height())
            QMouseEvent.accept()
        elif Qt.LeftButton and self._bottom_drag:
            self.resize(self.width(), QMouseEvent.pos().y())
            QMouseEvent.accept()
        elif Qt.LeftButton and self._corner_drag:
            self.resize(QMouseEvent.pos().x(), QMouseEvent.pos().y())
            QMouseEvent.accept()
        elif Qt.LeftButton and self._move_drag:
            self.move(QMouseEvent.globalPos() - self.move_DragPosition)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self._move_drag = False
        self._corner_drag = False
        self._bottom_drag = False
        self._right_drag = False

    def my_Qss(self):
        qssStyle = '''
                   QWidget#widget{
                   border-image:url(:/head.png);
                   }
                   QWidget#widget_2{
                   border-left:0.5px solid lightgray;
                   border-right:0.5px solid lightgray;
                   border-bottom:0.5px solid #e5e5e5;
                   border-bottom-left-radius: 5px;
                   border-bottom-right-radius: 5px;
                   padding:5px 5px 5px 5px
                   }
                   QWidget#widget_3{
                   background-color:#ffffff;
                   border-left:0.5px solid lightgray;
                    border-right:0.5px solid lightgray;
                   border-bottom:0.5px solid #e5e5e5;
                   border-bottom-left-radius: 5px;
                   border-bottom-right-radius: 5px;
                   padding:5px 5px 5px 5px
                   }
                   QWidget#widget_41{
                   background-color:#ffffff;
                   border-left:0.5px solid lightgray;
                    border-right:0.5px solid lightgray;
                   border-bottom:0.5px solid #e5e5e5;
                   border-bottom-left-radius: 5px;
                   border-bottom-right-radius: 5px;
                   padding:5px 5px 5px 5px
                   }
                   QWidget#widget_42{
                   background-color:#ffffff;
                   border-left:0.5px solid lightgray;
                   border-right:0.5px solid lightgray;
                   border-bottom:0.5px solid #e5e5e5;
                   border-bottom-left-radius: 5px;
                   border-bottom-right-radius: 5px;
                   padding:5px 5px 5px 5px
                   }
                   QWidget#page_T1{
                   background:#ffffff;
                   border:1px solid #CCCCCC;
                   border-radius:6px
                   }
                   QWidget#widget_5{
                   background-color:#ffffff;
                   border-left:0.5px solid lightgray;
                   border-right:0.5px solid lightgray;
                   border-bottom:0.5px solid #e5e5e5;
                   border-bottom-left-radius: 5px;
                   border-bottom-right-radius: 5px;
                   padding:5px 5px 5px 5px
                   }
                   QPushButton#pushButton
                   {
                   font-family:"Webdings";
                   text-align:top;
                   background:#6DDF6D;border-radius:5px;
                   border:none;
                   font-size:13px;
                   }
                   QPushButton#pushButton:hover{background:green;}
                   QPushButton#pushButton_2
                   {
                   font-family:"Webdings";
                   background:#F7D674;border-radius:5px;
                   border:none;
                   font-size:13px;
                   }
                   QPushButton#pushButton_2:hover{background:yellow;}
                   QPushButton#pushButton_3
                   {
                   font-family:"Webdings";
                   background:#F76677;border-radius:5px;
                   border:none;
                   font-size:13px;
                   }
                   QPushButton#pushButton_3:hover{background:red;}
                   QPushButton#pushButton_01
                   {
                   font-family:"Microsoft YaHei UI";
                   background:#6DDF6D;border-radius:6px;
                   color:#ffffff;
                   border:none;
                   font-size:10px;
                   font-weight:500;
                   }
                   QPushButton#pushButton_01:hover{background:red;}
                   '''
        self.setStyleSheet(qssStyle)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = IDAP()
    myWin.show()
    sys.exit(app.exec_())
