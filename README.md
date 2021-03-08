# drive_detect
使用pyqt制作能多驾驶员行为进行分类的程序
# 环境
python   
pyqt   
Qt Designer  
pyinstaller  
onnxruntime
# 介绍
该程序实现对图片、视频或者视频流中驾驶员行为进行检测分类   
imgs/文件夹存放图片
ui/文件夹存放ui文件，可以通过Qt Designer编辑  
videos/文件夹存放示例视频
weights/文件夹存放训练好的模型
# 运行
python drive.py
# 打包
pyinstaller -F drive.py  -w -i ./imgs/xm.ico  
注意打包后，需将资源文件放到对应文件夹  
