# Anime model

✅ 我们添加了 [*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth), 它是专门为动漫图像优化的而且模型大小更小

- [How to Use](#How-to-Use)
  - [PyTorch Inference](#PyTorch-Inference)
  - [ncnn Executable File](#ncnn-Executable-File)
- [Comparisons with waifu2x](#Comparisons-with-waifu2x)
- [Comparisons with Sliding Bars](#Comparions-with-Sliding-Bars)

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>

The following is a video comparison with sliding bar. You may need to use the full-screen mode for better visual quality, as the original image is large; otherwise, you may encounter aliasing issue.

https://user-images.githubusercontent.com/17445847/131535127-613250d4-f754-4e20-9720-2f9608ad0675.mp4

如何使用

### PyTorch 推断

预训练模型: [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

```bash
# 下载模型
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P experiments/pretrained_models
# 推断
python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth --input inputs
```

### ncnn 可执行文件

下载最新的移动版 [Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-macos.zip) **可执行文件 for Intel/AMD/Nvidia GPU**.

以Windows为例, 执行:

```bash
./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesrgan-x4plus-anime
```

## 与 waifu2x 模型比较

我们将Real-ESRGAN-anime模型与[waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)作了比较. 我们使用 `-n 2 -s 4` 作为 waifu2x 的参数.

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_2.png">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_3.png">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_4.png">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_5.png">
</p>

## 与Sliding Bars的比较

The following are video comparisons with sliding bar. You may need to use the full-screen mode for better visual quality, as the original image is large; otherwise, you may encounter aliasing issue.

https://user-images.githubusercontent.com/17445847/131536647-a2fbf896-b495-4a9f-b1dd-ca7bbc90101a.mp4

https://user-images.githubusercontent.com/17445847/131536742-6d9d82b6-9765-4296-a15f-18f9aeaa5465.mp4
