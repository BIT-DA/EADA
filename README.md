 ---

<div align="center">    
 
# Active Learning for Domain Adaptation: An Energy-based Approach

[Binhui Xie](https://binhuixie.github.io), [Longhui Yuan](https://scholar.google.com/citations?user=fVnEIZEAAAAJ&hl=en&oi=sra), [Shuang Li](https://shuangli.xyz), [Chi Harold Liu](https://scholar.google.com/citations?user=3IgFTEkAAAAJ&hl=en), [Xinjing Cheng](https://scholar.google.com/citations?user=8QbRVCsAAAAJ&hl=en) and [Guoren Wang](https://scholar.google.com.hk/citations?hl=en&user=UjlGD7AAAAAJ)


[![Paper](http://img.shields.io/badge/paper-arxiv.2112.01406-B31B1B.svg)](https://arxiv.org/abs/2112.01406)&nbsp;&nbsp;
[![Bilibili](https://img.shields.io/badge/Video-Bilibili-%2300A1D6?logo=bilibili&style=flat-square)](https://www.bilibili.com/video/BV1qa411h7Xm/?share_source=copy_web&vd_source=2536293932098e7a347341a231b3fb8b)&nbsp;&nbsp;
[![Slides](https://img.shields.io/badge/Poster-Dropbox-%230061FF?logo=dropbox&style=flat-square)](https://www.dropbox.com/s/8ozwc8uw1q1tqlf/eada_slides.pdf?dl=0)&nbsp;&nbsp;

</div>


Unsupervised domain adaptation (UDA) has recently emerged as an effective paradigm for generalizing deep neural networks to new target domains. However, there is still enormous potential to be tapped to reach the fully supervised performance. 

We start from an observation that energy-based models exhibit free energy biases when training (source) and test (target) data come from different distributions. Inspired by this inherent mechanism, we empirically reveal that a simple yet efficient energy-based sampling strategy sheds light on selecting the most valuable target samples than existing approaches requiring particular architectures or computation of the distances. 

Our algorithm, Energy-based Active Domain Adaptation (EADA), queries groups of target data that incorporate both domain characteristic and instance uncertainty into every selection round. Meanwhile, by aligning the free energy of target data compact around the source domain via a regularization term, domain gap can be implicitly diminished. 

![UDA over time](docs/eada.png)

Through extensive experiments, we show that EADA surpasses state-of-the-art methods on well-known challenging benchmarks with substantial improvements, making it a useful option in the open world.

For more information on EADA, please check our **[[Paper](https://arxiv.org/pdf/2112.01406.pdf)]**.

If you find this project useful in your research, please consider citing:

```bib
@inproceedings{xie2022active,
  title={Active learning for domain adaptation: An energy-based approach},
  author={Xie, Binhui and Yuan, Longhui and Li, Shuang and Liu, Chi Harold and Cheng, Xinjing and Wang, Guoren},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={8},
  pages={8708--8716},
  year={2022}
}
```


##  Setup Environment

For this project, we used python 3.7.5. We recommend setting up a new virtual environment:

**Step-by-step installation**

```bash
conda create --name activeDA -y python=3.7
conda activate activeDA

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

pip install -r requirements.txt
```

### Setup Datasets
- Download [The Office-31 Dataset](https://faculty.cc.gatech.edu/~judy/domainadapt/)
- Download [The Office-Home Dataset](http://hemanthdv.org/OfficeHome-Dataset/)
- Download [The VisDA-2017 Dataset](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

The data folder should be structured as follows:
```
├── data/
│   ├── office31/	
|   |   ├── amazon/
|   |   ├── dslr/
|   |   ├── webcam/	
│   ├── home/     
|   |   ├── Art/
|   |   ├── Clipart/
|   |   ├── Product/
|   |   ├── RealWorld/
│   ├── visda2017/
|   |   ├── train/
|   |   ├── validation/
│   └──	
```

Symlink the required dataset
```
ln -s /path_to_office31_dataset data/office31
ln -s /path_to_home_dataset data/home
ln -s /path_to_visda2017_dataset/clf/ data/visda2017
```

## Running the code

For Office-31
```
python main.py --cfg configs/office.yaml
```

For Office-Home
```
python main.py --cfg configs/home.yaml
```

For VisDA-2017
```
python main.py --cfg configs/visda2017.yaml
```

## Acknowledgements

This project is based on the following open-source projects. We thank their authors for making the source code publicly available.
- [Transferable-Query-Selection](https://github.com/thuml/Transferable-Query-Selection)

## Contact

If you have any problem about our code, feel free to contact

- [binhuixie@bit.edu.cn](mailto:binhuixie@bit.edu.cn)

or describe your problem in Issues.
