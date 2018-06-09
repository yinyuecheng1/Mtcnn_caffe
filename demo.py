import sys
sys.path.append('.')
sys.path.append('/home/cmcc/caffe-master/python')
import caffe
import cv2
import numpy as np
from Detector import Detector
import time 
import argparse

deploy = './Pnet/Pnet.prototxt'
caffemodel = './Pnet/Pnet.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = './Rnet/Rnet.prototxt'
caffemodel = './Rnet/Rnet.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)


deploy = './Onet/Onet.prototxt'
caffemodel = './Onet/Onet.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)

caffeNet = [net_12, net_24, net_48]

detector = Detector(net=caffeNet, 
                 min_face_size=24,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 slide_window=False)

'''
usage:  python test.py --filename=test5.jpg

'''
parser = argparse.ArgumentParser(description='Test mtcnn', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', dest='filename', help='Image to process', default='test4.jpg', type=str)
args = parser.parse_args()


filename = args.filename
img = cv2.imread(filename)
print '####imsize%%', img.shape
t1 = time.time()

boxes, boxes_c = detector.detect_pnet(img)
boxes, boxes_c = detector.detect_rnet(img, boxes_c)
boxes, boxes_c = detector.detect_onet(img, boxes_c)

print('time: %.3fs' %(time.time() - t1))
print('num of boxes:', boxes_c.shape[0])

if boxes_c is not None:
    draw = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b in boxes_c:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
        cv2.putText(draw, '%.3f' % b[4], (int(b[0]), int(b[1])), font, 0.4, (255, 255, 255), 1)
    while True:
        cv2.imshow("detection result", draw)
    	#f = filename.split('.')
    	#cv2.imwrite(''.join([f[:-1], "_annotated.", f[-1]]), draw)
    	if cv2.waitKey(1) & 0xFF == ord('q'):
            print "I'm done"
            break


