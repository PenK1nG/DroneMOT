<!--
 * @Author: PenK1nG 578068089@qq.com
 * @Date: 2024-12-12 14:33:13
 * @LastEditors: PenK1nG 578068089@qq.com
 * @LastEditTime: 2024-12-14 17:28:49
 * @FilePath: /DroneMOT/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# DroneMOT

This repository contains the official implementation of the paper:
**"DroneMOT: Drone-based Multi-Object Tracking Considering Detection Difficulties and Simultaneous Moving of Drones and Objects"** by Peng Wang, Yongcai Wang, and Deying Li.

---

## Overview
DroneMOT introduces a multi-object tracking system tailored for drone-based applications, addressing unique challenges such as detection difficulties and simultaneous movement of both drones and objects. This repository provides the necessary code, models, and instructions to reproduce the results and extend the work.

---

## Getting Started
Follow the steps below to set up your environment and prepare for running DroneMOT.

### 1. Environment Setup

#### Step 1.1: Install PyTorch and Related Libraries
Ensure that you have **Conda** installed. Then, install the specified version of PyTorch (1.7.0) and related libraries:

```bash
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
```

#### Step 1.2: Clone the FairMOT Repository
DroneMOT builds upon FairMOT for tracking. Clone the repository and navigate to its root directory:

```bash
git clone https://github.com/ifzhang/FairMOT.git
cd FairMOT
```

#### Step 1.3: Install Required Python Packages
Install the required Python packages for FairMOT:

```bash
pip install cython
pip install -r requirements.txt
```

---

### 2. Building Dependencies
DroneMOT requires specific custom packages to be built. Follow the steps below to set them up.

#### Step 2.1: Build DCNv2

DCNv2 (Deformable Convolutional Networks) is used within FairMOT. Clone and compile the compatible version for PyTorch 1.7:

```bash
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```

##### Alternative Option for DCNv2
If you encounter compatibility issues due to a different PyTorch version, consider using the following alternative implementation:

- Repository: [CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2)

Follow their instructions to build the package.

#### Step 2.2: Build the Correlation Package

The correlation package is a custom component required by DroneMOT. To build it, follow these steps:

1. Navigate to the DroneMOT root directory:

   ```bash
   cd ${DroneMOT_ROOT}
   ```

2. Build the correlation package:

   ```bash
   cd ./src/lib/models/networks/correlation_package
   python setup.py build_ext --inplace
   ```

---

## Todo
We are actively improving DroneMOT. Below is a list of ongoing tasks:


- [x] **Complete the overall training code** (✔️ Completed)
- [ ] **Enhance the tracking code** (⏳ In Progress)

Stay tuned for updates as we refine and expand the repository's functionality.
