import argparse         # 写命令行参数用的
import cv2              # OpenCV
import glob             # 整理文件名用的
import os               # 操作系统类
import subprocess       # 目前不知道有什么用，肯定是跟操作系统交互的
from ffmpeg import video# 处理视频的模块
# RRDBNet,用于人脸强化
from basicsr.archs.rrdbnet_arch import RRDBNet
# RealESRGAN的类
from realesrgan import RealESRGANer

def main():
    # 处理传递的参数
    parser = SetArgs()
    args = parser.parse_args()                  # argsy用来存放参数
    # ----- <实例化模型> -----
    upsampler = RealESRGANer(
        scale=args.netscale,
        model_path=args.model_path,
        model=args.model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        half=args.half
    )
    # ----- </实例化模型> -----
    # ----- <处理视频> -----
    os.makedirs(args.inputName + '_Images', exist_ok=True)
    if(video.video_trans_img(args.inputFolder + '/' + args.inputName, args.output  + '/' + args.inputName + '/' + 'Images')):
        pass
    # ----- </处理视频> -----
    pass


def SetArgs():
    target = argparse.ArgumentParser()
    target.add_argument('--inputName', type=str, help='要处理的视频文件的名字')
    target.add_argument('--inputFolder', type=str, default='inputs/Video', help='视频文件在项目中的目录,默认为inputs/Video')
    target.add_argument('--output', type=str, default='results/Video', help='输出视频的存放目录,默认为results/Video')
    target.add_argument('--netscale', type=int, default=4,help='神经网络升采样的倍数')
    target.add_argument('--model_path', type=str, default='experiments/pretrained_models/RealESRGAN_x4plus.pth', help='模型的路径，默认使用GANx4plus')
    target.add_argument('--outscale', type=float, default=4, help='视频最终采用的升采样倍数')
    target.add_argument('--fps', type=int, default=24, help='视频处理使用的帧数标准，默认为24帧/每秒')
    target.add_argument('--face_enhance', action='store_true', help='使用GFPGAN增强人脸')
    target.add_argument('--half', action='store_true', help='推断处理时是否使用半精度')
    target.add_argument('--block', type=int, default=23, help='RRDB网络中的num_block参数')
    target.add_argument(
        '--alpha_upsampler',
        type=str,
        default='reaesrgan',
        help='Alpha通道使用的升采样器，使用realesrgan或者bicubic')
    target.add_argument('--tile', type=int, default=0, help='tile的大小，默认为0即测试过程中没有tile')
    target.add_argument('--tile_pad', type=int, default=10, help='tile填充')
    return target

def vid2img(args=argparse.Namespace):
    if(video.video_trans_img(args)):
        pass
    pass

def readIMG():
    ret = cv2.imread(os.path, cv2.IMREAD_UNCHANGED)
    # 判断图片的通道情况
    if len(ret.shape) == 3 and ret.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None
    return ret, img_mode

if(__name__ == '__main__'):
    main()