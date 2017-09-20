from models import object_detection
from config import config
import cv2

model_name = config.models["1"]
net = object_detection.Net(graph_fp='%s/frozen_inference_graph.pb' % model_name,
                           labels_fp='data/label.pbtxt',
                           num_classes=90,
                           threshold=0.6)
CAMERA_MODE = 'camera'
STATIC_MODE = 'static'


def demo(mode=CAMERA_MODE):
    if mode == STATIC_MODE:
        img = 'test_images/1.jpg'
        net.predict(img=cv2.imread(img))
        cv2.waitKey()
        cv2.destroyAllWindows()
    elif mode == CAMERA_MODE:
        cap = cv2.VideoCapture(0)

        while True:

            ret, frame = cap.read()
            net.predict(img=frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    demo(mode=STATIC_MODE)
