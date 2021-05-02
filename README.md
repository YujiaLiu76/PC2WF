# PC2WF
by Yujia Liu, Stefano D'Aronco, Konrad Schindler, Jan Dirk Wegner.

### Introduction
This repository is for our ICLR2021 paper '[PC2WF: 3D WIREFRAME RECONSTRUCTION FROM RAW POINT CLOUDS](https://arxiv.org/pdf/2103.02766.pdf)'.

### Installation
This code relies on [FCGF](https://github.com/chrischoy/FCGF) as backbone network. Please make sure that you installed all requirements.

This code has been tested with CUDA 10.0, Python 3.7, Pytorch 1.2.0, MinkowskiEngine 0.2.9.

### Data Preparation

1. Clone this repository.
    ```
    git clone https://github.com/YujiaLiu76/PC2WF
    cd PC2WF
    ```

1. Put pointcloud dataset into directory `abc_data/clean/xyz/`. 
Put corresponding groudtruch files into directory `abc_data/clean/gt/`. (Please refer to the examples in those directories we put into)

2. Add noise to clean pointclouds. The default sigma and clip values are both 0.01.
    ```
    cd gen_data
    python noise_addnoise.py
    ```

3. Generate path dataset for training and evaluation.
    ```
    cd ..
    python noise_gen_patch_straight.py
    ```

4. Train a model.
    ```
    python main.py -d abc_data -p 50 -nt 0.01 -lpt 0.01 -lnt 0.01 -s 0.01 -c 0.01
    ```
    Please refer to `main.py` for detailed explanation of arguments. 
    (We have provided a pretained model with default arguments on abc dataset.)

5. Visualize results.
    We provided scripts for visualizing predicted wireframe results from pointclouds. The following scripts will read metadata generated in `abc_data/pathches_*/test/` and visualize the predicted wireframes. Note that, they will use pretained models as default.
    - predict vertexes and edges
    ```
    cd visualize
    python run_test_line.py
    ```
    - visualize
    ```
    python visualize_line.py
    ```
