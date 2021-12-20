# SALAD: Self-supervised Aggregation Learning for Anomaly Detection on X-Rays

This repository contains the pytorch implementation of the proposed method *SALAD: Self-supervised Aggregation Learning for Anomaly Detection on X-Rays* which has been accepted for MICCAI 2020.

### Citation

You find the PDF of *SALAD: Self-supervised Aggregation Learning for Anomaly Detection on X-Rays* MICCAI 2020 paper [here](https://link.springer.com/content/pdf/10.1007%2F978-3-030-59710-8_46.pdf).

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
python main.py CXR_author unet ../log/salad ../data/author
```

### Test

Run the following command:

```
python test.py CXR_author unet ../log/salad_test ../data/author --load_model ../log/salad/model_round10.tar
```

You can find our trained model checkpoints [here](https://drive.google.com/drive/folders/11XQLKhcrEllQ-hzCxrAkAESiuhdEqFIX?usp=sharing).

### License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
