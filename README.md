# PIPNet: Pixel-in-Pixel Net

## Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild

### Notes

* Used datasets are `300W` or `300W+WFLW`
* Download pretrained weights for training, see `Pretrained weights`
* Trained was runed on 4 GPUs (be careful with accumulation)
* Train code was modified to detect only eye landmarks, but it can be easily returned to the default version

### Dataset Overview

|   Datasets   |   Train   |  Test   |  LMK   |      Resolution       |
| :----------: | :-------: | :-----: | :----: | :-------------------: |
|     AFW      |    337    |    -    |   68   |   300x400 - 700x900   |
|     IBUG     |     -     |   135   |   68   |   300x400 - 500x600   |
|     LFPW     |    811    |   224   |   68   |   250x250 - 600x600   |
|    HELEN     |   2000    |   330   |   68   |   300x400 - 500x600   |
|     WFLW     |   10000   |    -    |   98   |   300x300 - 800x800   |
| 🔥**Merged**🔥 | **13148** | **689** | **68** | **250x250 - 800x900** |

### Conda Env Installation

```bash
conda create --name PIPNet python=3.11 -y
conda activate PIPNet
pip install opencv-python timm onnx onnxruntime-gpu onnx-simplifier onnxoptimizer scipy matplotlib PyYAML tqdm loguru icecream
```

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh 4 --train --test --test-onnx --epochs 120 --batch-size 32 --input-size 256
```

### Test

```bash
python main.py --test --test-onnx --exp <expID|PATH.pt>
```

### Demo

```bash
python demo.py
```

### Results

| Backbone | Epochs |  NME  | NME (300W)* | NME (300W+WFLW)* |                                 Pretrained weights                                 |
| :------: | :----: | :---: | :---------: | :--------------: | :--------------------------------------------------------------------------------: |
| IRNet18  |  120   | 3.27  |   3.2585    |      3.2705      | [model](https://github.com/jahongir7174/PIPNet/releases/download/v0.0.1/IR18.pth)  |
| IRNet50  |  120   | 3.11  |   3.1746    |      3.1127      | [model](https://github.com/jahongir7174/PIPNet/releases/download/v0.0.1/IR50.pth)  |
| IRNet100 |  120   | 3.08  |   3.1458    |      3.0923      | [model](https://github.com/jahongir7174/PIPNet/releases/download/v0.0.1/IR100.pth) |

![Alt Text](./assets/demo.gif)

### 300W Dataset Preparation

1. **Download** [`300W`](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) datasets

2. **Unzip the data** into the structure below:

   ```text
   300W
     ├── afw
     ├── helen
     ├── ibug
     └── lfpw
   ```

3. **Run the below command** for dataset preparation:

   ```bash
   python -c 'from utils import util; util.DataGenerator(data_dir="300W", target_size=256).run()'
   ```

### Datasets

* [300W+WFLW](https://github.com/jahongir7174/PIPNet/releases/download/v0.0.1/LMK.zip) - load post-processed dataset for training.

* [`300W`](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) - dataset includes (AFW, IBUG, LFPW, HELEN), offers a good mix of diverse faces and conditions:
  
  * `AFW` - Annotated Faces in the Wild dataset is small in size, but AFW is known for its difficult cases, including faces with extreme poses and occlusions. This makes it ideal for evaluating how well your model performs in difficult, real-world situations.

  * `IBUG` - dataset is specifically used as a challenging test set. It includes images with large pose variations, occlusions, and difficult lighting conditions. This makes it ideal for evaluating how well your model performs in difficult, real-world situations.

  * `LFPW` - Labeled Face Parts in-the-Wild offers a wide variety of faces collected from the web in terms of lighting, expressions, and occlusions, which helps in generalizing your model to different real-world scenarios.

  * `HELEN` - provides high-resolution images with detailed facial landmark annotations, making it a good candidate for training models that require precise landmark detection.

* [`WFLW`](https://wywu.github.io/projects/LAB/WFLW.html) - Wider Facial Landmark in-the-Wild dataset is a comprehensive and challenging dataset for 98 facial landmark detection. It was introduced to address more difficult cases in facial landmark detection, such as large pose variations, heavy occlusions, and extreme expressions.

* [`COFW`](https://data.caltech.edu/records/bc0bf-nc666) - Caltech Occluded Faces in the Wild dataset is designed to be more challenging than earlier datasets with 68 facial landmarks by focusing on occlusions by objects (e.g., glasses, hats, hands) or other faces.

* [`AFLW`](https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw) - Annotated Facial Landmarks in the Wild dataset is a widely used resource in the field of facial landmark detection and face analysis. It was designed to provide a large, diverse collection of faces with annotated 21 facial landmarks, covering a wide range of poses, expressions, and occlusions. AFLW is excellent for training robust facial landmark detection models that need to generalize well across different poses and conditions.

* [`LaPa`](https://github.com/jd-opensource/lapa-dataset) - Labeled Faces Parsing in the Wild dataset was designed for facial parsing, which involves segmenting different regions of the face (like eyes, nose, mouth, etc.), as well as for 106 facial landmark detection. It provides detailed annotations that are useful for training models to understand both the structure and specific regions of the face.

### Drowsiness Development Notes

|  EXP  | Epochs |  BS   |  LR   |  NME   |           NOTE            |
| :---: | :----: | :---: | :---: | :----: | :-----------------------: |
|  000  |  120   |  16   | 5E-4  | 1.7903 |         300W-WFLW         |
|  001  |  120   |  32   | 5E-4  | 1.7646 |         300W-WFLW         |
|  002  |  120   |  64   | 5E-4  | 1.7590 |         300W-WFLW         |
|  003  |  120   |  128  | 5E-4  | 1.7902 |         300W-WFLW         |
|  014  |  120   |  256  | 5E-4  | 1.7844 |         300W-WFLW         |
|  004  |  120   |  16   | 5E-4  | 1.8076 |      300W-WFLW-42dot      |
|  005  |  120   |  32   | 5E-4  | 1.7809 |      300W-WFLW-42dot      |
|  006  |  120   |  64   | 5E-4  | 1.7903 |      300W-WFLW-42dot      |
|  007  |  120   |  128  | 5E-4  | 1.7538 |      300W-WFLW-42dot      |
|  013  |  120   |  256  | 5E-4  | 1.7726 |      300W-WFLW-42dot      |
|  008  |  120   |  16   | 5E-4  | 1.8249 |   300W-WFLW-42dot-ReLu    |
|  009  |  120   |  32   | 5E-4  | 1.7997 |   300W-WFLW-42dot-ReLu    |
|  010  |  120   |  64   | 5E-4  | 1.8086 |   300W-WFLW-42dot-ReLu    |
|  011  |  120   |  128  | 5E-4  | 1.7981 |   300W-WFLW-42dot-ReLu    |
|  012  |  120   |  256  | 5E-4  | 1.7819 |   300W-WFLW-42dot-ReLu    |
|  015  |  120   |  256  | 5E-4  | 1.7819 | 300W-WFLW-42dot-ITG-ReLu  |
|  016  |  120   |  256  | 5E-4  | 1.7726 | 300W-WFLW-42dot-ITG-PReLu |
|  000  |  120   |  256  | 5E-4  | 1.9700 | 300W-WFLW-42dot-REGR-ReLu |
|  181  |  120   |  256  | 5E-4  | 1.9469 | 300W-WFLW-42dot-PHA-REGR-ReLu |
|  184  |  120   |  256  | 5E-4  | 1.8799 | 300W-WFLW-42dot-PHA-REGR-ReLu-RegnetX800 |


```text
TARGET:
    NME = 1.7%

REFERENCES 300W-WFLW
    Epoch=120   deep=18     NME=3.2685  EXP=018     ***   4-GPUs
    Epoch=120   deep=50     NME=3.1148  EXP=050     ***   4-GPUs
    Epoch=120   deep=100    NME=3.0838  EXP=100     ***   4-GPUs

REIMPLEMENTATION 300W-WFLW
    DEFAULT:
    Epoch=120   deep=18     NME=3.2705  EXP=0       ***   4-GPUs
    Epoch=120   deep=50     NME=3.1127  EXP=1       ***   4-GPUs
    Epoch=120   deep=100    NME=3.0923  EXP=2       ***   4-GPUs

    reg: 1.0 = 1.7969
    reg: 2.5 = 1.7889
    reg: 1.8 = 1.7833
    reg: 1.5 = 1.7789
    reg: 2.0 = 1.7712   !!!

EYES MODIFICATION:  # Implementation: Defaul <----> Drowsiness
    args.yaml
        num_nb: 5                                   # number of neighbors   ERRORS: [0, 12]-train | [1, 2]-test | failed
        num_lms: 12                                 # number of landmarks
        selected_index: *flip_index_eyes_anchor     # [flip_index_anchor, flip_index_eyes_anchor]

RECOMMENDED:
    1. Use deep=18              --> light and show almost similar performance to heavy models
    2. Use 12 eyes landmarks    --> drowsiness implementation
    3. Use 5 neighbors          --> each eye has 6 landmarks
    4. Use reg: 2.0             --> more focus on coordinate regression


Architectures:
    MobileNet-V2
        PT-NME:     1.8334
        ONNX-NME:   1.8332

    MobileNet-V3-Large
        PT-NME:     1.7922
        ONNX-NME:   1.7926

    MobileOne-S1
        PT-NME:     1.8151
        ONNX-NME:   1.8142


Mean-lmk:
300W-WFLF-lmk12 = 300W
    1.7712
    1.7714

300W-WFLF-lmk12 = 300W-WFLF
    1.7737
    1.7736

300W-WFLF-lmk12 = 300W-WFLF-lmk12
    1.7737
    1.7736

300W-WFLF-lmk12-42dot = 300W
    1.8005
    1.8014

300W-WFLF-lmk12-42dot = 300W-WFLF
    1.828
    1.8282

300W-WFLF-lmk12-42dot = 300W-WFLF-lmk12-42dot
    1.8134
    1.813
```

### TODO

* [ ] **GSSL architecture**

### References

* [**arXiv article: Pixel-in-Pixel Net**](https://arxiv.org/abs/2003.03771)
* [**PIPNet - Original**](https://github.com/jhb86253817/PIPNet)
* [**PIPNet - Jahongir**](https://github.com/jahongir7174/PIPNet)
* [**PIPNet - Shohruh**](https://github.com/Shohruh72/PIPNet)
* [**PIPNet - Shohruh-V2**](https://github.com/Shohruh72/LitePIPNet)
