import argparse 
from traffic_counter import TrafficCounter

#GROUP_3
#HASEEB_AHMAD_FA17_BCS_125
#ARQAM_SHAKEEL_FA17_BCS_001

def CLI():
    #Define default values here to make documentation self-updating
    minArea_default       = 200
    direction_default     = ['H']




    parser = argparse.ArgumentParser(description='Finds the contours on a video file')
    parser.add_argument('-p','--path',type=str)
    parser.add_argument('-a','--minArea',type=int,default=minArea_default)
    parser.add_argument('-d','--direction', type=str,default=direction_default)
    args = parser.parse_args()
    return args

def main(args):
    video_source   = args.path
    line_direction = args.direction
    min_area       = int(args.minArea)
    tc = TrafficCounter(video_source,
                        line_direction,
                        min_area,)

    tc.main_loop()

if __name__ == '__main__':
    args = CLI()
    main(args)