# ERN

This repository contains the implementation of our paper, ["ERN: Edge Loss Reinforced Semantic Segmentation Network for Remote Sensing Images"](http://www.mdpi.com/2072-4292/10/9/1339).

In this repository you will find:
- ERN prototxt and caffemodel;
- Python codes for training and testing;
- C++ code (VS2013 + OpenCV3.0) for simple shadow detection; 

## Requirement and Usage

### 1. [Caffe](https://github.com/BVLC/caffe)
  - Please follow Yu Liu's instruction and install the modified version of Caffe from following link: https://github.com/Walkerlikesfish/CaffeGeo.git. 
  - Notice that if you use Caffe, please cite their papers.

### 2. Train
  - Prepare the data;
  - Make sure the **file path** in solver.prototxt & train.prototxt & run_training.py is correct;
  - Run python run_training.py in terminal.
  
### 3. Test（inference）
  - Prepare the data;(In our experiments, we first splited the original image into 256\*256 )
  - Make sure the **file path** in infer.prototxt & VH_infer.py & UAV_infer.py is correct;
  - Run python VH_infer.py in terminal.
  
### 4. Stitch the patches
  - We have split the images using https://github.com/Walkerlikesfish/HSNRS/tree/master/script/data_preprocessing
    - Make sure the **file path** in VH_assemble.py & UAV_assemble.py is correct;
    - Run python VH_assemble.py in terminal.
    
### 5. Evaluate the semantic segmentation performance in shadow-affected regions
  - Shadow detection for ISPRS Vaihingen Dataset (Windows)
    - VS2013 + OpenCV3.0
    - The contrast preserving decolorization has been used. 
  - Evaluation (Linux)
    - Make sure the **file path** in testperformance.py (**def xf_set_test_vh()**)is correct;
    - Run python testperformance.py (**def xf_set_test_vh()**) in terminal.

  
## License and Citation
Please cite the following paper if you find the project helpful to your research.

	@article{Liu2018ERN,
	title={ERN: Edge Loss Reinforced Semantic Segmentation Network for Remote Sensing Images},
	author={Liu, Shuo and Ding, Wenrui and Liu, Chunhui and Liu, Yu and Li, Hongguang},
	journal={Remote Sensing},
	volume={10},
	number={9},
	pages={1339},
	year={2018}
	}

This code is shared under a Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) Creative Commons licensing.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
