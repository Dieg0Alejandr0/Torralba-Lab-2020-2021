# Description

The following repository consists of a data pipeline of randomly sampling 2D images of 3D rendering from Facebook's Replica dataset (see render.cpp) done appropriately for 
StyleGAN and ProgressiveGAN training and creating interpolation movies from StyleGANs trained on said data and visualizing the paths of said movies in 2D space. Note that 
render.cpp is a modified script from Facebook's Replica repository, so all rights to said script go to Facebook .Inc and their affliates. Many thanks to my mentor Jonas Wulff
for being a helpful guide during my time as a UROP at the Torralba Lab. 


## How To Run!
As mentioned above, please use "render.cpp" to produce a needed 2D dataset for GAN training. The produced model snapshot can be then used in "Interpolation.ipynb" to produce the needed model analysis for scene understanding. Please refer to the libraries/packages listed below for any software requirements for running the code. Please consult the bibliographic references to learn more about the GANs used and the LPIPS metric!

## Bibliographic References

[ProgressiveGAN](https://arxiv.org/pdf/1710.10196.pdf), 
[StyleGAN](https://arxiv.org/pdf/1812.04948.pdf),
[LPIPS](https://arxiv.org/pdf/1801.03924.pdf)

## Relevant Dataset(s)

[Facebook's Replica](https://github.com/facebookresearch/Replica-Dataset)

## Software Libraries/Packages Required

C++: 
[Replica](https://github.com/facebookresearch/Replica-Dataset),
[EGL](https://github.com/facebookresearch/Replica-Dataset),
[Pangolin](https://github.com/facebookresearch/Replica-Dataset)


Python:
[PyTorch](https://pytorch.org/),
[numpy](https://numpy.org/),
[LPIPS](https://github.com/richzhang/PerceptualSimilarity),
[scikit-learn](https://scikit-learn.org/stable/),
[tqdm](https://github.com/tqdm/tqdm),
[PIL](https://pillow.readthedocs.io/en/stable/),
[imageio](https://imageio.readthedocs.io/en/stable/),
[matplotlib](https://matplotlib.org/)
