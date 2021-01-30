# Vehicle Counter
 This is a semester project of 7th semester of DIP (Digital Image Processing) developed in python. It works on a video and counts the number of vehicle passing
 through a line in the centre of screen. It works on blop technique and uses Open Cv for it.
 
You need these libraries installed in your system.
OpenCv 4.4.0
Python 3.7.8
Numpy 1.19.1

How to run using console:
python main.py --path visiontraffic.avi -d h
OR
python main.py --path visiontraffic.avi -d v

Here the arguments are:
--path visiontraffic.avi  ----It is path of video.
-d h or v or H or V   ---- This is direction of green line on screen. Horizontal or vertical.
