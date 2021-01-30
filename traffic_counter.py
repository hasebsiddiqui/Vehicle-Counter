import numpy as np
import cv2

#GROUP_3
#HASEEB_AHMAD_FA17_BCS_125
#ARQAM_SHAKEEL_FA17_BCS_001

#python main.py --path visiontraffic.avi -d h
class TrafficCounter(object):
    def __init__(self,video_source,
                 line_direction='H',
                 min_area = 200,):
        self.p1_count_line     = None
        self.p2_count_line     = None
        self.counter           = 0
        self.line_direction    = line_direction       
        self.line_position     = 0.5
        self.minArea           = min_area
        self.video_source      = cv2.VideoCapture(video_source)
        self.prev_centroids    = []          #this will contain the coordinates of the centers in the previous
        self._vid_height = None
        self._vid_width = None

        #--Getting frame dimensions 
        self._compute_frame_dimensions()
        self._set_up_line(line_direction,0.5)
        self.collage_frame = self._create_collage_frame()

    def _set_up_line(self,line_direction,line_position):
        if line_direction.upper()=='H' or line_direction is None:
            fract = int(self._vid_height*float(line_position))
            self.p1_count_line = (0,fract)
            self.p2_count_line = (self._vid_width,fract)
        elif line_direction.upper() == 'V':
            fract = int(self._vid_width*float(line_position))
            self.p1_count_line = (fract,0)
            self.p2_count_line = (fract,self._vid_height)
        else:
            raise ValueError('Expected an "H" or a "V" only for line direction')

    def _create_collage_frame(self):
        self.collage_width  = self._vid_width  * 2
        self.collage_height = self._vid_height * 2
        collage_frame = np.zeros((self.collage_height,self.collage_width,3),dtype=np.uint8)
        return collage_frame

    def _compute_frame_dimensions(self):
        grabbed,img = self.video_source.read()
        while not grabbed:
            grabbed,img = self.video_source.read()

        self._vid_height = img.shape[0]
        self._vid_width = img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.raw_avg = np.float32(img)



    def _draw_bounding_boxes(self,frame,bounding_points,currentX,currentY,prev_cx,prev_cy):
        cv2.drawContours(frame,[bounding_points],0,(0,255,0),1)
        # line between last position and current position
        cv2.line(frame,(prev_cx,prev_cy),(currentX,currentY),(0,0,255),1)
        cv2.circle(frame,(currentX,currentY),3,(0,0,255),4)


    def _is_line_crossed(self,frame,currentX,currentY,prev_cx,prev_cy):
        is_crossed = False
        if self.line_direction.upper() == 'H':
            if (prev_cy <= self.p1_count_line[1] <= currentY) or (currentY <= self.p1_count_line[1] <= prev_cy):
                self.counter += 1
                cv2.line(frame,self.p1_count_line,self.p2_count_line,(0,255,0),5)
                is_crossed = True

        elif self.line_direction.upper() == 'V':
            if (prev_cx <= self.p1_count_line[0] <= currentX) or (currentX <= self.p1_count_line[0] <= prev_cx):
                self.counter += 1
                cv2.line(frame,self.p1_count_line,self.p2_count_line,(0,255,0),5)
                is_crossed = True
        return is_crossed

    def bind_objects(self,frame,thresh_img):

        #CV_RETR_EXTERNAL
        #CV_RETR_LIST
        #CV_RETR_TREE
        #CHAIN_APPROX_SIMPLE
        #CHAIN_NONE
        cnts, hierarchy = cv2.findContours(thresh_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cur_centroids  = []
        for c in cnts:
            # ignore contours that are smaller than this area
            if cv2.contourArea(c) < self.minArea:
                continue

            #How to do it in numpy. Calculate box coordinates.
            rect   = cv2.minAreaRect(c)
            points = cv2.boxPoints(rect)
            points = np.int0(points)


            #Getting the center coordinates of the contour box
            currentX = int(rect[0][0])
            currentY = int(rect[0][1])

            w,h = rect[1]   #Unpacks the width and height of the frame

            C = np.array((currentX,currentY))

            # print(rect)
            # print("DD")
            # print(points)
            #print("gg")
            #print(C)

            cur_centroids.append(C)

            #Finding the centroid of c in the previous frame
            if len(self.prev_centroids)==0: 
                prev_cx,prev_cy = currentX,currentY
            elif len(cnts)==0: 
                prev_cx,prev_cy = currentX,currentY
            else:
                minPoint = None
                minDist = None
                for i in range(len(self.prev_centroids)):
                    dist = np.linalg.norm(C - self.prev_centroids[i])                #numpy's way to find the euclidean distance between two points
                    if (minDist is None) or (dist < minDist):
                        minDist = dist
                        minPoint = self.prev_centroids[i]
                #This if is meant to reduce overcounting errors
                if minDist < w/2:
                    prev_cx,prev_cy = minPoint
                else:
                    prev_cx,prev_cy = currentX,currentY
            
            _is_crossed = self._is_line_crossed(frame,currentX,currentY,prev_cx,prev_cy)
            if _is_crossed:
                print(f"Total Count: {self.counter}")
            self._draw_bounding_boxes(frame,points,currentX,currentY,prev_cx,prev_cy)


        self.prev_centroids = cur_centroids       #updating centroids for next frame



    def make_collage_of_four(self,up_left_img,up_right_img,down_left_img,down_right_img):
        middle_width  = self._vid_width
        middle_height = self._vid_height
        total_height  = self.collage_frame.shape[0]
        total_width   = self.collage_frame.shape[1]
        if len(up_left_img.shape) < 3:
            up_l = cv2.cvtColor(up_left_img,cv2.COLOR_GRAY2BGR)
        else:
            up_l = up_left_img
        
        if len(up_right_img.shape) < 3:
            up_r = cv2.cvtColor(up_right_img,cv2.COLOR_GRAY2BGR)
        else:
            up_r = up_right_img

        if len(down_left_img.shape) < 3:
            down_l = cv2.cvtColor(down_left_img,cv2.COLOR_GRAY2BGR)
        else:
            down_l = down_left_img

        if len(down_right_img.shape) < 3:
            down_r = cv2.cvtColor(down_right_img,cv2.COLOR_GRAY2BGR)
        else:
            down_r = down_right_img

        self.collage_frame[0:middle_height,0:middle_width]                      = up_l    #setting up_left image
        self.collage_frame[0:middle_height,middle_width:total_width]            = up_r 
        self.collage_frame[middle_height:total_height,0:middle_width]           = down_l
        self.collage_frame[middle_height:total_height,middle_width:total_width] = down_r

    def main_loop(self):
        #weight of next frame in average
        rate_of_influence = 0.01

        while True:
            grabbed,img = self.video_source.read()
            if not grabbed:
                break
            #--------------#get current frame index
            frame_id = int(self.video_source.get(1))
            #current frame
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            originalFrame = img.copy()

            cv2.accumulateWeighted(img,self.raw_avg,rate_of_influence)

            # reference background average image
            background_avg = cv2.convertScaleAbs(self.raw_avg)
            subtracted_img = cv2.absdiff(background_avg,img)

            ##-------Adding extra bluring to remove unnecessary details------
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)

            ##-------Applying threshold
            ## original image, threshold value, max, min
            _,threshold_img  = cv2.threshold(subtracted_img,30,255,0)


            ##-------Drawing bounding boxes and counting
            # Giving frame 3 channels for color (for drawing colored boxes)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            self.bind_objects(img,threshold_img)

            ##-------Showing Frames
            cv2.imshow('Original Frame',originalFrame)              #working_img is the frame after being cropped and masked
            cv2.imshow('Background-Subtracted',subtracted_img)  #subtracted_img is the frame after the background has been subtracted from it
            cv2.imshow('Threshold Applied',threshold_img)         #threshold_img is threshold_img plus the noise reduction functions
            cv2.imshow('Running Avg of Background',background_avg)
            cv2.line(img,self.p1_count_line,self.p2_count_line,(0,0,255),1)   #counting line

            self.font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,f"Total cars: {self.counter}",(15,self._vid_height-15),self.font,1,(255,255,255),3)
            cv2.imshow('Motion Detection',img)

            self.make_collage_of_four(subtracted_img,background_avg,threshold_img,img)
            cv2.putText(self.collage_frame,f"Frame: {frame_id}",(15,self.collage_height-15),self.font,1,(255,255,255),3)
            cv2.imshow('Traffic Counter',self.collage_frame)

            ##-------Termination Conditions
            k = cv2.waitKey(25) & 0xFF
            if k == 27 or k == ord('q') or k == ord('Q'):
                break
        cv2.destroyAllWindows()