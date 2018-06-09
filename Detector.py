import cv2
#import mxnet as mx
import time
import image_processing
#from mx.model import FeedForward
import numpy as np
from nms import py_nms

class Detector(object):
    def __init__(self,
                 net,
                 min_face_size=24,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 slide_window=False):

        self.Pnet, self.Rnet, self.Onet = net[0], net[1], net[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window
    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel
                input image, channels in BGR order here
            scale: float number
                scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        img_resized = image_processing.transform(img_resized)
        return img_resized  # (batch_size, c, h, w)
    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox


    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment(offenset)
        Returns:
        -------
            bboxes after refinement
        """

	bbox_c = bbox.copy() #(x1, y1, x2, y2, confidence) , 
        w = bbox[:, 2] - bbox[:, 0] + 1 # (N, )
        w = np.expand_dims(w, 1) #(N, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1) #(N, 1)
        reg_m = np.hstack([w, h, w, h]) # (N, 4)
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c  #(N, 5)
    def generate_bbox(self, map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            map: numpy array , n x m x 1(n x m)
                detect score for each position
            reg: numpy array , n x m x 4(4 x n x m )
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
	stride = 2
        cellsize = 12

        t_index = np.where(map > threshold) # a tuple contains 2 array:(?, ?)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)] # (?, ) respectively.

        reg = np.array([dx1, dy1, dx2, dy2]) #(4, ?)
        score = map[t_index[0], t_index[1]] # (?, )
        boundingbox = np.vstack([np.round((stride*t_index[1])/scale),
                                 np.round((stride*t_index[0])/scale),
                                 np.round((stride*t_index[1]+cellsize)/scale),
                                 np.round((stride*t_index[0]+cellsize)/scale),
                                 score,
                                 reg]) # vertical cancatenate,  (4+1+4, ?)

        return boundingbox.T # (?, 9)
    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image(that is:a image generate from 0)
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image( that is:input image)
            ex, ey : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
	tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1,  bboxes[:, 3] - bboxes[:, 1] + 1 #(N, )
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box, )), np.zeros((num_box, )) #(N, )
        edx, edy = tmpw.copy()-1, tmph.copy()-1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1) # box over the right bound.
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0) # box over the left bound.
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size    # find initial scale
        im_resized = self.resize_image(im, current_scale)
        _, _, current_height, current_width = im_resized.shape

        if self.slide_window:
            # sliding window
            temp_rectangles = list()
            rectangles = list()     # list of rectangles [x11, y11, x12, y12, confidence] (corresponding to original image)
            all_cropped_ims = list()
            while min(current_height, current_width) > net_size:
                current_y_list = range(0, current_height - net_size + 1, self.stride) if (current_height - net_size) % self.stride == 0 \
                    else range(0, current_height - net_size + 1, self.stride) + [current_height - net_size]   # cancatenate list.
                current_x_list = range(0, current_width - net_size + 1, self.stride) if (current_width - net_size) % self.stride == 0 \
                    else range(0, current_width - net_size + 1, self.stride) + [current_width - net_size]

                for current_y in current_y_list:
                    for current_x in current_x_list:
                        cropped_im = im_resized[:, :, current_y:current_y + net_size, current_x:current_x + net_size]

                        current_rectangle = [int(w * float(current_x) / current_width), int(h * float(current_y) / current_height),
                                             int(w * float(current_x) / current_width) + int(w * float(net_size) / current_width),
                                             int(h * float(current_y) / current_height) + int(w * float(net_size) / current_width),
                                             0.0]
                        temp_rectangles.append(current_rectangle)
                        all_cropped_ims.append(cropped_im)
              	current_scale *= self.scale_factor
              	im_resized = self.resize_image(im, current_scale)
                _, _, current_height, current_width = im_resized.shape

            all_cropped_ims = np.vstack(all_cropped_ims)
            N, _, _, _ = all_cropped_ims.shape

	    self.Pnet.blobs['data'].reshape(N, 3, 12, 12)
     	    self.Pnet.blobs['data'].data[...] = all_cropped_ims
            out = self.Pnet.forward()
	    cls_prob = out['prob1'][:,1,0,0] # (N, ), prob of face.
   	    reg = out['conv4-2'][:,:] # (N, 4), offenset of 4 cordinates.
	    keep_inds = np.where(cls_prob > self.thresh[0])[0] # (N, ), index of box(or window img) which scores is upper the threshold.

	     
	    if len(keep_inds) > 0:
		rect = []
		for ind in keep_inds:
		    rect.append(temp_rectangles[ind])
                #boxes = np.vstack(temp_rectangles[ind] for ind in keep_inds)  # (N, 5)
		boxes = np.vstack(rect)
                boxes[:, 4] = cls_prob[keep_inds]
                reg = reg[keep_inds].reshape(-1, 4)
	    else:
	        return None, None
	    
	    keep = py_nms(boxes, 0.7, 'Union')
	    boxes = boxes[keep]
	    
	    boxes_c = self.calibrate_box(boxes, reg[keep])
	   
        else:
	    # FCN
	    all_boxes = list()
            while min(current_height, current_width) > net_size:
	        self.Pnet.blobs['data'].reshape(1, 3, current_height, current_width)
	        self.Pnet.blobs['data'].data[...] = im_resized
	        out = self.Pnet.forward()
                cls_map = out['prob1'] # (1, 2, ?, ?)
	        reg = out['conv4-2'] # (1, 4, ?, ?)
                boxes = self.generate_bbox(cls_map[0, 1, :, :], reg[0, :, :, :], current_scale, self.thresh[0])

                current_scale *= self.scale_factor
                im_resized = self.resize_image(im, current_scale)
                _, _, current_height, current_width = im_resized.shape

                if boxes.size == 0:
                    continue
                keep = py_nms(boxes[:, :5], 0.5, 'Union')
                boxes = boxes[keep]
                all_boxes.append(boxes)

            if len(all_boxes) == 0:
                return None, None

            all_boxes = np.vstack(all_boxes)

            # merge the detection from first stage
            keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
            all_boxes = all_boxes[keep]
            boxes = all_boxes[:, :5]

            bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
            bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

            # refine the boxes
            boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                                 all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                                 all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                                 all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                                 all_boxes[:, 4]]) 
            boxes_c = boxes_c.T

        return boxes, boxes_c # (N, 5)
    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
	if dets is None:
            return None, None
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting RNet batch size
        batch_size = self.rnet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print("You may need to reset RNet batch size if this info appears frequently, \)
face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            cropped_ims[i, :, :, :] = image_processing.transform(cv2.resize(tmp, (24, 24)))

	self.Rnet.blobs['data'].reshape(num_boxes, 3, 24, 24)
	self.Rnet.blobs['data'].data[...] = cropped_ims
	out = self.Rnet.forward()
    	cls_scores = out['prob1'] #(N, 2)
    	reg = out['bbox_fc'] #(N, 4)
        cls_scores = cls_scores[:, 1].flatten() #(N,)
        keep_inds = np.where(cls_scores > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        keep = py_nms(boxes, 0.7)
        boxes = boxes[keep]

        boxes_c = self.calibrate_box(boxes, reg[keep])

        return boxes, boxes_c

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        if dets is None:
            return None, None
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting ONet batch size
        batch_size = self.onet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print("You may need to reset ONet batch size if this info appears frequently, \)
face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            cropped_ims[i, :, :, :] = image_processing.transform(cv2.resize(tmp, (48, 48)))

	
	self.Onet.blobs['data'].reshape(num_boxes, 3, 48, 48)
	self.Onet.blobs['data'].data[...] = cropped_ims	 
	out = self.Onet.forward()
    	cls_scores = out['prob1']
    	reg = out['conv6-2']
    	pts_prob = out['conv6-3']

        cls_scores = cls_scores[:, 1].flatten()
        keep_inds = np.where(cls_scores > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        boxes_c = self.calibrate_box(boxes, reg)

        keep = py_nms(boxes_c, 0.7, "Minimum")
        boxes_c = boxes_c[keep]

        return boxes, boxes_c    #(N, 5)





