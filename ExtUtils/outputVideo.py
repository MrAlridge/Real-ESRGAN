import argparse         # 写命令行参数用的
import cv2              # OpenCV
import glob             # 整理文件名用的
import os               # 操作系统类
import subprocess       # 把视频转换成一帧一帧的图片
# RRDBNet,用于人脸强化
from basicsr.archs.rrdbnet_arch import RRDBNet
# RealESRGAN的类
from realesrgan import RealESRGANer

def main():
    # 处理传递的参数
    parser = argparse.ArgumentParser()
    SetArgs(parser)


def SetArgs(target = argparse.ArgumentParser()):
    target.add_argument('--inputName', type=str, help='要处理的视频文件的名字')
    target.add_argument('--inputFolder', type=str, default='inputs/Video', help='视频文件在项目中的目录,默认为inputs/Video')
    target.add_argument('--output', type=str, default='results/Video', help='输出视频的存放目录,默认为results/Video')
    target.add_argument('--netscale', type=int, default=4,help='神经网络升采样的倍数')
    target.add_argument('--outscale', type=float, default=4, help='视频最终采用的升采样倍数')
    target.add_argument('--fps', type=int, default=24, help='视频处理使用的帧数标准，默认为24帧/每秒')
    target.add_argument('--face_enhance', action='store_true', help='使用GFPGAN增强人脸')
    target.add_argument('--half', action='store_true', help='推断处理时是否使用半精度')