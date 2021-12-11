# -----<summary>-----
# 这个文件是一个工具集合，用来实现输入内容和输出内容
# -----</summary>-----

from _typeshed import Self
import os
import glob
from ffmpeg import video

TASK_DIR = './Videos'

class VideoTools:
    videoName = str
    def ProcessVid2Pic(vidName = videoName):
        ret = video.video_trans_img(TASK_DIR + vidName, TASK_DIR + '/OutPutImg')
        return ret
    def initDir():                  # 初始化目录
        os.mkdir('OutPutImg')
    def __init__(self, Path, vidName) -> None:
        self.videoName = vidName                # 需要处理的视频的名字
        if(os.curdir != TASK_DIR):
            os.chdir(TASK_DIR)
        self.initDir()
        try:
            if(self.ProcessVid2Pic(self.videoName) == False):
                raise FailedVid2ImgException
        except FailedVid2ImgException as e:
            print('未能成功将视频转换为图片')
        # ----- 这里接入Real-ESRGAN -----
        pass

class FailedVid2ImgException(Exception):
    def __init__(self) -> None:
        pass