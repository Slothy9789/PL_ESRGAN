# <div align="center">PLanetary Enhanced Super-Resolution Generative Adversarial Network (PL-ESRGAN)


 <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

基于 https://github.com/xinntao/Real-ESRGAN 的 PL-ESRGAN 的目标是开发出**实用的行星影像修复算法，特别关注月球、火星和水星**。

如果 PL-ESRGAN 对你有帮助，可以给本项目一个 Star :star: ，或者推荐给你的朋友们，谢谢！:blush: <br/>

---

## :wrench: 依赖以及安装

- Python >= 3.9
- [PyTorch >= 2.3](https://pytorch.org/)

#### 安装

1. 把项目克隆到本地

2. 安装各种依赖

    ```bash
    pip install basicsr
    pip install numpy
    pip install opencv-python
    pip install Pillow
    pip install torchvision
    pip install tqdm
    ```

3. 将项目内的 `/basicsr` 复制并覆盖虚拟环境中新安装的basicsr文件夹，路径可参考如下 `~/envs/xxx/lib/python3.9/site-packages/basicsr`

## :european_castle: 模型库

月球：`~:\PL_ESRGAN\weights\net_g_Moon.pth`

火星：`~:\PL_ESRGAN\weights\net_g_Mars.pth`

水星：`~:\PL_ESRGAN\weights\net_g_Mercury.pth`

## :zap: 快速上手

```bash
python .\inference_realesrgan.py -n model -i inputs -s4.0 -o output
```

其中，`model` 是已经训练好的月球/火星/水星超分模型，`inputs`是待超分的数据，超分结果存放在指定的`output`文件夹中
