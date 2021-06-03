import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget
import threading


# 点击运行程序闪退，使用cmd运行查看错误信息
##打包该程序到一个文件夹
# pyinstaller drive.py --noconsole
# pyinstaller  drive.py  -w -i ./imgs/uestc.ico --add-data=./imgs;imgs --add-data=./ui;ui --add-data=./weights;weights --add-data=./videos;videos
# 打包指定图标 --icon="logo.ico"
# pyinstaller drive.py --noconsole -i ./imgs/xm.ico
# 资源文件也需要添加到dist对应文件夹
# 打包该程序到一个exe文件 不显示控制台，指定图标仍然需要把资源文件放到dist对应文件夹
# pyinstaller -F  -w -i ./imgs/xm.ico drive.py
# 打包该程序到一个exe文件 不显示控制台，指定图标，添加资源文件到exe
# pyinstaller -F  drive.py -w -i ./imgs/xm.ico --add-data="imgs;."
# 还是必须复制资源文件夹
# pyinstaller -F  -w -i ./imgs/xm.ico --add-data=./imgs;imgs --add-data=./ui;ui --add-data=./weights;weights drive.py
# 错误ImportError: numpy.core.multiarray failed to import 可能是numpy版本原因，卸载重装numpy即可

# Python threading模块提供Event对象用于线程间通信。用于主线程控制其他线程的执行，事件主要提供了四个方法wait、clear、set、isSet
#
# set()：可设置Event对象内部的信号标志为True
# clear()：可清除Event对象内部的信号标志为False
# isSet()：Event对象提供了isSet()方法来判断内部的信号标志的状态。当使用set()后，isSet()方法返回True；当使用clear()后，isSet()方法返回False
# wait()：该方法只有在内部信号为True的时候才会被执行并完成返回。当内部信号标志为False时，则wait()一直等待到其为True时才返回
class drive(QWidget):

    def __init__(self):
        super().__init__()
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = uic.loadUi("./ui/drive.ui")
        # 加载图片
        self.img_path = "./imgs/uestc.jpg"
        self.ui.img_path.setText(self.img_path)
        self.img = QPixmap(self.img_path).scaled(640, 480)
        self.ui.img_label.setPixmap(self.img)
        # 加载视频路径
        # self.cap_path = "rtmp://58.200.131.2:1935/livetv/hunantv"
        self.cap_path = "./videos/3_23_1_s.mp4"
        self.ui.cap_path.setText(self.cap_path)
        # 加载模型
        self.modelpath = r"weights/mobilenetv2_224_acc=85.6154.onnx"
        self.net = cv2.dnn.readNetFromONNX(self.modelpath)
        self.label_name = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
        # self.label_name = ["正常", "未定义", "无人", "分心", "抽烟", "使用手机", "喝水", "抓痒", "拿东西"]

        self.ui.choose_btn.clicked.connect(self.choose)
        self.ui.predict_btn.clicked.connect(self.predict)

        self.ui.choose_video_btn.clicked.connect(self.choose_video)
        self.ui.play_btn.clicked.connect(self.play)
        self.ui.stop_btn.clicked.connect(self.stop)
        self.ui.stop_btn.setEnabled(False)
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    def inference(self, src):
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        image = cv2.resize(src, (224, 224))
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))

        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
        self.net.setInput(blob)
        probs = self.net.forward()
        index = np.argmax(probs)
        return index

    def choose(self):
        # print("choose")
        # options = QFileDialog.Options()
        # fname, _ = QFileDialog.getOpenFileName(self, "Select labels", "", "Images (*.png *.xpm *.jpg)", options=options)
        # # fname = QFileDialog.getOpenFileName(self, '打开文件', './')
        # fileName, fileType = QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
        #                                                  "All Files(*);;Text Files(*.txt)")
        # print(fileName)

        img_path, _ = QFileDialog.getOpenFileName(self, '选择图片', './imgs', "Images (*.png *.xpm *.jpg)")
        self.ui.img_path.setText(img_path)
        self.img_path = img_path
        self.img = QPixmap(img_path).scaled(640, 480)
        self.ui.img_label.setPixmap(self.img)
        self.ui.predict_label.setText("未预测")
        # print(fileType)

    def choose_video(self):
        cap_path, _ = QFileDialog.getOpenFileName(self, '选择视频', './videos', "Videos (*.mp4 *.avi *.fiv)")
        self.ui.cap_path.setText(cap_path)
        self.cap_path = cap_path
        self.stop()
        self.ui.img_label.setPixmap(self.img)
        # self.set_first_frame()

    # def set_first_frame(self):
    #     cap = cv2.VideoCapture(self.cap_path)
    #     while cap.isOpened():
    #         success, frame = cap.read()
    #         if not success:
    #             print("Can't receive frame (stream end?). Exiting ...")
    #             break
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #         frame = cv2.resize(frame, (640, 480))
    #         img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    #         self.ui.img_label.setPixmap(QPixmap.fromImage(img))
    #
    #     cap.release()

    def play(self):
        print("play")
        video_t = threading.Thread(target=self.display, args=(self.cap_path,self.ui.img_label))
        # 设置为守护进程防止意外退出，进程不停止
        video_t.setDaemon(True)
        video_t.start()
        self.ui.stop_btn.setEnabled(True)
        self.ui.play_btn.setEnabled(False)
        self.stopEvent.clear()

    def stop(self):
        print("stop")
        self.stopEvent.set()
        self.ui.stop_btn.setEnabled(False)
        self.ui.play_btn.setEnabled(True)

    def display(self, url, label):
        """显示"""
        cap = cv2.VideoCapture(url)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            index = self.inference(frame)
            self.ui.predict_label.setText(self.label_name[index])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
            cv2.waitKey(1)

            # 判断关闭事件是否已触发
            if self.stopEvent.is_set():
                # # 关闭事件置为未触发，清空显示label
                # self.ui.img_label.setPixmap(self.img)
                break
        cap.release()
        #退出程序还原
        self.ui.img_label.setPixmap(self.img)
        self.ui.stop_btn.setEnabled(False)
        self.ui.play_btn.setEnabled(True)

    def predict(self):
        print("predict!")
        self.stop()
        src = cv2.imread(self.img_path)
        index = self.inference(src)
        self.ui.predict_label.setText(self.label_name[index])


if __name__ == '__main__':
    app = QApplication([])
    # 设置程序图标
    app.setWindowIcon(QIcon('./imgs/uestc.jpg'))
    drive = drive()
    drive.ui.show()
    app.exec_()
