
from deepface import DeepFace
import threading
import cv2
import dlib

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class OpcvCapture(threading.Thread):
    def __init__(self, win_name, cam_name):
        super().__init__()
        self.cam_name = cam_name
        self.win_name = win_name

    def run(self):
        # capture = cv2.VideoCapture(self.cam_name)
        capture = cv2.VideoCapture('test.mp4')#说明：参数0表示默认为笔记本的内置第一个摄像头。


        while (True):
            # 按帧读取视频帧
            # ret,frame是获取cap.read()方法的两个返回值。
            # 其中ret是布尔值，如果读取帧是正确的则返回True， 如果文件读取到结尾，它的返回值就为False。
            # frame就是每一帧的图像，是个三维矩阵。
            ret, frame = capture.read()
            #取灰度
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
            ffaces = detector(gray, 0)
            #人脸数
            faces = detector(frame)
            #检测到人脸
            if ret is True:
                b, g, r = cv2.split(frame)  # 分离色道 opencv读入的色道是B,G,R
                fame2 = cv2.merge([r, g, b])  # 合成R,G,B
                dets = detector(fame2, 1)  # 使用detector进行人脸检测

            for k, d in enumerate(ffaces):
                # 用红色矩形框出人脸
                #cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
                # 计算人脸热别框边长
                self.face_width = d.right() - d.left()
                obj = DeepFace.analyze(img_path=frame, actions=['age', 'emotion'])
                age = obj[0]['age']
                emo = obj[0]['dominant_emotion']
                cv2.putText(frame, str(age), (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            4)
                cv2.putText(frame, str(emo), (d.left(), d.bottom() + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            4)

            for i, face in enumerate(dets):  # 遍历返回值 index是几个人脸
                shape = predictor(fame2, face)  # 寻找人脸的68个脸

                for index, pt in enumerate(shape.parts()):  # 遍历所有的点，把点用蓝色的圈圈表示出来
                    pt_pox = (pt.x, pt.y)
                    cv2.circle(frame, pt_pox, 2, (255, 0, 0), 2)

                   #展示帧
            cv2.imshow("me", frame)

            # 等待键盘输入，通常用来判断是否退出
            # waitKey（）方法本身表示等待键盘输入，参数是1，表示延时1ms切换到下一帧图像，对于视频而言。
            k = cv2.waitKey(5)
            if k & 0xff == ord('q'):
                break

if __name__ == "__main__":
    camera1 = OpcvCapture("Face", 0)
    camera1.start()