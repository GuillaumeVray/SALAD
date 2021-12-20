# SALAD: Self-supervised Aggregation Learning for Anomaly Detection on X-Rays

This repository contains the pytorch implementation of the proposed method *SALAD: Self-supervised Aggregation Learning for Anomaly Detection on X-Rays* which has been accepted for MICCAI 2020.

### Citation

You find the PDF of *SALAD: Self-supervised Aggregation Learning for Anomaly Detection on X-Rays* MICCAI 2020 paper [here](https://cibm.ch/wp-content/uploads/Bozorgtabar2020_Chapter_SALADSelf-supervisedAggregatio.pdf).

If you use our code or find our work relevant to your research, please cite the paper as follows:

```
@inproceedings{bozorgtabar2020salad,
  title={Salad: Self-supervised aggregation learning for anomaly detection on x-rays},
  author={Bozorgtabar, Behzad and Mahapatra, Dwarikanath and Vray, Guillaume and Thiran, Jean-Philippe},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={468--478},
  year={2020},
  organization={Springer}
}
```

### Setup

Execute the following bash script:

```
bash setup.sh
```

It downloads the data and moves it to the `data` folder. It also creates a conda environment called `SALAD`, containing the necessary dependencies to run the code. Activate the conda environment with `conda activate SALAD`.

### Train

Run the following command:

```
python main.py CXR_author CXR_resnet18 ../log/CXR_test ../data/author --lr 0.0001 --n_epochs 30 --lr_milestone 100 --batch_size 32 --weight_decay 1e-6 --n_jobs_dataloader 12 --isize 224 --rep_dim 200 --w_rec 0.5 --w_svdd 0.5
```

### Test

Run the following command:

```
python test.py CXR_author unet ../log/finaltest/ ../data/author --batch_size 32 --n_jobs_dataloader 12 --isize 256 --k 150 --rep_dim 200 --load_model ../log/test4/model_round10.tar
```

You can find our trained model checkpoints [here](https://drive.google.com/drive/folders/11XQLKhcrEllQ-hzCxrAkAESiuhdEqFIX?usp=sharing).

### License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
