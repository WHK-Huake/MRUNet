# MRUNet
Multi-Scale Retinex Unfolding Network for Low-Light Image Enhancement.

Our peper has been accepted by TMM. 

## Train
Download LOL dataset from (https://daooshee.github.io/BMVC2018website/) and change the dataset path in `training.yaml`, and then run

```
python train.py
```


## Test
test LOL dataset. Change the testdataset path of `test.py` and then run

```
python test.py
```

or unpair dataset run

```
python test_unpair.py
```

## Citation
If you find the code helpful in your research or work, please cite the following paper:
```
@ARTICLE{10891570,
  author={Wang, Huake and Hou, Xingsong and Li, Jutao and Yan, Yadi and Sun, Wenke and Zeng, Xin and Zhang, Kaibing and Cao, Xiangyong},
  journal={IEEE Transactions on Multimedia}, 
  title={Multi-Scale Retinex Unfolding Network for Low-Light Image Enhancement}, 
  year={2025},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TMM.2025.3543015}}

```
