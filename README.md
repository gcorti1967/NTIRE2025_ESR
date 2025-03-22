# [NTIRE 2025 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

<div align=center>
<img src="https://github.com/Amazingren/NTIRE2025_ESR/blob/main/figs/logo.png" width="400px"/> 
</div>

## News
- :t-rex: February 8th, 2025: Our Challenge Repo. is ready!


## About the Challenge

In collaboration with the NTIRE workshop, we are hosting a challenge focused on Efficient Super-Resolution ([NTIRE2025_ESR](https://codalab.lisn.upsaclay.fr/competitions/21620)). This involves the task of enhancing the resolution of an input image by a factor of x4, utilizing a set of pre-existing examples comprising both low-resolution and their corresponding high-resolution images. The challenge encompasses one :trophy: main track which consists of three :gem: sub-tracks, i.e., the Inference Runtime, FLOPs (Floating Point Operations Per Second), and Parameters. The baseline method in NTIRE2025_ESR is [EFDN](https://arxiv.org/pdf/2204.08759) (*Wang Yan, 2023*), the 1st place for the overall performance of NTIRE2023 Efficient Super-Resolution Challenge. Details are shown below:

- :trophy: Main-track: **Overall Performance** (Runtime, Parameters, FLOPs,) the aim is to obtain a network design / solution with the best overall performance in terms of inference runtime, FLOPS, and parameters on a common GPU (i.e., NVIDIA RTX A6000 GPU) while being constrained to maintain or improve the PSNR results.

- :gem: Sub-track 1: **Inference Runtime**, the aim is to obtain a network design / solution with the lowest inference time (runtime) on a common GPU (i.e., NVIDIA RTX A6000 GPU) while being constrained to maintain or improve over the baseline method EFDN in terms of number of parameters, FLOPs, and the PSNR result.

- :gem: Sub-track 2: **FLOPs**, the aim is to obtain a network design / solution with the lowest amount of FLOPs on a common GPU (i.e., NVIDIA RTX A6000 GPU) while being constrained to maintain or improve the inference runtime, the parameters, and the PSNR results of EFDN.

- :gem: Sub-track 3: **Parameters**, the aim is to obtain a network design / solution with the lowest amount of parameters on a common GPU (i.e., NVIDIA RTX A6000 GPU) while being constrained to maintain the FLOPs, the inference time (runtime), and the PSNR result of EFDN.

It's important to highlight that to determine the final ranking and challenge winners, greater weight will be given to teams or participants who demonstrate improvements in more than one aspect (runtime, FLOPs, and parameters) over the provided reference solution.

To ensure fairness in the evaluation process, it is imperative to adhere to the following guidelines:
- **Avoid Training with Specific Image Sets:**
    Refrain from training your model using the validation LR images, validation HR images, or testing LR images. The test datasets will not be disclosed, making PSNR performance on the test datasets a crucial factor in the final evaluation.

- **PSNR Threshold and Ranking Eligibility:**
    Methods with a PSNR below the specified threshold (i.e., 26.90 dB on DIV2K_LSDIR_valid and, 26.99 dB on DIV2K_LSDIR_test) will not be considered for the subsequent ranking process. It is essential to meet the minimum PSNR requirement to be eligible for further evaluation and ranking.


## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

- Step1: install Pytorch first:
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

- Step2: install other libs via:
```pip install -r requirements.txt```

or take it as a reference based on your original environments.

## The Validation datasets
After downloaded all the necessary validate dataset ([DIV2K_LSDIR_valid_LR](https://drive.google.com/file/d/1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj/view?usp=sharing) and [DIV2K_LSDIR_valid_HR](https://drive.google.com/file/d/1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v/view?usp=sharing)), please organize them as follows:

```
|NTIRE2025_ESR_Challenge/
|--DIV2K_LSDIR_valid_HR/
|    |--000001.png
|    |--000002.png
|    |--...
|    |--000100.png
|    |--0801.png
|    |--0802.png
|    |--...
|    |--0900.png
|--DIV2K_LSDIR_valid_LR/
|    |--000001x4.png
|    |--000002x4.png
|    |--...
|    |--000100x4.png
|    |--0801x4.png
|    |--0802x4.png
|    |--...
|    |--0900.png
|--NTIRE2025_ESR/
|    |--...
|    |--test_demo.py
|    |--...
|--results/
|--......
```

## How to test the baseline model?

1. `git clone https://github.com/gcorti1967/NTIRE2025_ESR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 56
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.

## References
If you feel this codebase and the report paper is useful for you, please cite our challenge report:
```
@inproceedings{ren2024ninth,
  title={The ninth NTIRE 2024 efficient super-resolution challenge report},
  author={Ren, Bin and Li, Yawei and Mehta, Nancy and Timofte, Radu and Yu, Hongyuan and Wan, Cheng and Hong, Yuxin and Han, Bingnan and Wu, Zhuoyuan and Zou, Yajun and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6595--6631},
  year={2024}
}
```

## Organizers
- Yawei Li (yawei.li@vision.ee.ethz.ch)
- Bin Ren (bin.ren@unitn.it)
- Hang Guo (cshguo@gmail.com)
- Zongwei Wu (zongwei.wu@uni-wuerzburg.de)
- Radu Timofte (Radu.Timofte@uni-wuerzburg.de) 

If you have any question, feel free to reach out the contact persons and direct managers of the NTIRE challenge.


## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 