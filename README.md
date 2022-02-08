# README

### 1 环境配置

#### 1.1 face_recognition

下载[dilb](https://pan.baidu.com/s/15bQ2vEU9pJUgwjNiQTNMmw) 解压码：whms

```makefile
pip install dlib-19.17.99-cp37-cp37m-win_amd64.whl
pip install face_recognition
```

#### 1.2 skimage

```makefile
conda install scikit-image
```

#### 1.3 tqdm

```
pip install tqdm
```

#### 1.4 scipy

在[这里]([Python Extension Packages for Windows - Christoph Gohlke (uci.edu)](https://www.lfd.uci.edu/~gohlke/pythonlibs/))找到包含MKL库版本的numpy和scipy文件（对应自己的py版本）下载并安装。

```
pip install numpy-1.21.5+mkl-cp37-cp37m-win_amd64.whl
pip install scipy-1.7.3-cp37-cp37m-win_amd64.whl
```

#### 1.5 pytorch

按照[官网](https://pytorch.org/)步骤安装即可。

#### 1.6 opencv

```
pip install opencv-python
```

#### **1.7** insightface & onnx

```makefile
pip install insightface==0.2.1 onnxruntime moviepy
pip install onnxruntime-gpu #推荐1.2.0版本
```

### 2 使用说明

将视频文件放于video文件夹下(可以使用`python movevideo.py`将指定目录下所有视频文件挪至video文件夹下)，使用下列命令开始采集。

```makefile
python face_cut.py -r 512 -t 300 -m 0 -i 5 -s 60 -b 100
```

`-r` ：采集出来人脸的分辨率

`-t` ：可以容忍的原人脸分辨率的范围（分辨率大于**r-t**的人脸会被采集并resize到r）

`-m` ：crop和align的方式，0-VGGface，1-FFHQ（默认VGG）

`-i` ：初始人脸采集间隔

`-s` ：未检测到人脸时跳过的帧数量

`-b` ：失焦模糊检测的阈值

其他：文件中两个参数Max_int = 40，Min_int = 5指动态采集间隔的最大和最小值，增加Min_int和Max_int会减少人脸采集数量，提高人脸图像之间的差异。反之，人脸采集数量增加，差异减小。

建议对部分视频进行试采集，根据结果使用调整各参数。

采集后的人脸对应的id会存于同目录下的json文件下。

### 3 视频选择条件

- 选择包含较多亚洲人脸的高帧率4k视频，这些视频可以从以下网站下载：

  - [website1](https://www.bugutv.cn/) 	[website2](https://www.gaoqingw.com/) 	[website3](https://gaoqing.fm/) 	[website4](http://www.yinfans.me/) 	[website5](https://www.mypianku.net/)	[website6](https://rarbgmirror.org/)
  - 注意下载时选择**4k分辨率**或者**原盘**的资源（符合要求的电影的大小基本**大于10GB**，格式多为**mkv**，电视剧类推）

- 主要人物的人脸采集数量应达到100张以上，若无法满足要求，可以调整采集参数或者放弃该视频

- 采集的参数最好设置为在达到要求数量情况下的最高值（减小`-t`或者增加`-b`），以减少后期人工筛选所需要的时间

- 采集时观察到采集的人脸出现部分模糊的现象，可以舍弃这部分样本（提高`-b`）或者使用[GFPGAN](https://github.com/TencentARC/GFPGAN)进行优化（配置好后使用提供的`inference_gfpgan.py `替换原文件）

  ```makefile
  python inference_gfpgan.py --upscale 1 --test_path DATASET_PATH --save_root SAVE_PATH
  ```

- 使用磨皮等美颜技术修改后的视频采集效果不佳，应少选择此类视频

- 推荐采集的视频类型：日/韩剧，亚洲电影和部分综艺等

- 避免选择政治敏感的视频

------

如有任何疑问，请与stau7001@sjtu.edu.cn联系。

