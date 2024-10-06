# <div align="center">PLanetary Enhanced Super-Resolution Generative Adversarial NetWork (PL-ESRGAN)

 <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

PL-ESRGAN aims at developing **Practical Algorithms for Planetary  Image Restoration**，specifically focusing on the **Moon, Mars, and Mercury**，based on  https://github.com/xinntao/Real-ESRGAN.<br>

🌌 If Real-ESRGAN is helpful, please help to ⭐ this repo or recommend it to your friends 😊 <br>

---

## 🔧 Dependencies and Installation

#### Dependencies

- Python >= 3.9
- PyTorch >= 2.3

#### Installation

1. Clone repo

1. Install dependent packages

    ```bash
    pip install basicsr
    pip install numpy
    pip install opencv-python
    pip install Pillow
    pip install torchvision
    pip install tqdm
    ```

1. Copy the `/basicsr` folder from the project and overwrite the newly installed `basicsr` folder in the virtual environment. You can refer to the following path: `~/envs/xxx/lib/python3.9/site-packages/basicsr`.


## :european_castle: Model Library

Moon:         `~:\PL_ESRGAN\weights\net_g_Moon.pth`

Mars:          `~:\PL_ESRGAN\weights\net_g_Mars.pth`

Mercury:    `~:\PL_ESRGAN\weights\net_g_Mercury.pth`

## :zap: Quick Start

```bash
python .\inference_realesrgan.py -n model -i inputs -s4.0 -o outputs
```

Where `model` refers to a pre-trained super-resolution model for the Moon, Mars, or Mercury, `inputs` are the images to be reconstructed at super-resolution, and the results are stored in the `outputs` folder.
