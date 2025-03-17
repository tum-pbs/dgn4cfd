<h2 align="center">DGN4CFD: Diffusion Graph Nets for Computational Fluid Dynamics</h2>

<h6 align="center">Official implementation of Diffusion Graph Nets from [<a href="https://openreview.net/pdf?id=uKZdlihDDn">📄 Research Paper</a>]</h6>

<h4 align="center">Learning Distributions of Complex Fluid  Simulations with Diffusion Graph Networks (ICLR2025 - ⭐ Oral ⭐)</h4>

<h6 align="center"><img src="https://i.ibb.co/NNvNGCz/pngfind-com-more-button-png-6671010.png" width="16"> <a href="https://ge.in.tum.de/about/mario-lino/">Mario Lino</a>, <img src="https://i.ibb.co/3hc9dTt/google-deepmind-ea6c104d05.webp" width="14"> <a href="https://tobiaspfaff.com/">Tobias Pfaff</a> and <img src="https://i.ibb.co/NNvNGCz/pngfind-com-more-button-png-6671010.png" width="16"> <a href="https://ge.in.tum.de/about/n-thuerey/">Nils Thuerey</a></h6>

<h6 align="center">
    <img src="https://i.ibb.co/NNvNGCz/pngfind-com-more-button-png-6671010.png" width="16"> Technical University of Munich
    <img src="https://i.ibb.co/3hc9dTt/google-deepmind-ea6c104d05.webp" width="16"> Google DeepMind
</h6>

## About 
Fluid flows, are often poorly represented by a single mean solution. For many practical applications, it is crucial to **access the full distribution of possible flow states**, from which relevant statistics (e.g., RMS and two-point correlations) can be derived. 
**Diffusion Graph Nets (DGNs)** enable direct sampling of these states via **flow matching** or **diffusion**-based denoising, given a **mesh** discretization of the system and its physical parameters. This allows for the **efficient computation of flow statistics** without running long and expensive numerical simulations.

<p align="center">
  <img src="https://i.ibb.co/G2JqcCN/DGN-Ellipse-Flow-compressed.gif"  width="800" />
</p>
<p align="center">
Velocity and pressure fields around an elliptical cylinder sampled from a DGN
</p>
<br>

<p align="center">
  <img src="https://i.ibb.co/DpNPLmm/dgn-wing.gif" width="400" />
</p>
<p align="center">
  Pressure field on a wing model sampled from a DGN
</p>
<br>

DGNs can also work on a **compressed latent mesh**. Operating in this latent space not only **reduces inference time** but also **mitigates the introduction of undesired high-frequency noise** in the solutions. We refer to these models as **Latent DGNs (LDGNs)**.
<br>
<br>

<p align="center">
  <img src="https://i.ibb.co/nmF8gvq/equilibrium2.png" width="900" />
</p>

(a) (L)DGNs learn the probability distribution of the systems' converged states provided only a short trajectory of length $\delta \ll T$ per system. (b) A preview of our turbulent wing experiment. The distribution learned by our LDGN model accurately captures the variance of all states (bottom right) despite seeing only an incomplete distribution for each wing during training (top right).

<br>

When **trained on short incomplete simulations** (which lack sufficient diversity to fully represent their individual flow statistics), (L)DGNs accurately capture full distributions where other methods, such as Gaussian mixture models or variational autoencoders (VAEs), suffer from noise and mode collapse.

<p align="center">
  <img src="https://i.ibb.co/nz094dX/pdf.png" width="600" />
</p>
<p align="center">
Probability density function from the DGN, LDGN, baseline models, and ground truth. The
DGN and LDGN show the best distributional accuracy.
</p>


For more results and detailed methods, check our [paper](https://openreview.net/pdf?id=uKZdlihDDn).


## Installation
Python 3.10 or higher and [PyTorch](https://pytorch.org/) 2.4 or higher are required.
We recommend installing **dgn4cfd** in a virtual environment, e.g., using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
It can installed from the current directory by running:

```bash
pip install -e .
```

This also installs [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) and compiles [PyTorch Cluster](https://github.com/rusty1s/pytorch_cluster), so it may take a while.
Once **dgn4cfd** has been installed, it can be imported in Python as follows:

```python
import dgn4cfd as dgn
```

## Models

**dgn4cfd** includes the following probabilistic models:
- Diffusion Graph Net (`dgn4cfd.nn.DiffusionGraphNet`)
- Latent Diffusion Graph Net (`dgn4cfd.nn.LatentDiffusionGraphNet`)
- Vanilla Graph Net (`dgn4cfd.nn.VanillaGnn`)
- Bayesian Graph Net (`dgn4cfd.nn.BayesianGnn`)
- Gaussian Mixture Graph Net (`dgn4cfd.nn.GaussianMixtureGnn`)
- Variational Graph Autoencoder (`dgn4cfd.nn.VGAE`)
- Flow-Matching Graph Net (`dgn4cfd.nn.FlowMatchingGraphNet`)
- Latent Flow-Matching Graph Net (`dgn4cfd.nn.LatentFlowMatchingGraphNet`)

Details are available in Appendix B of our [paper](https://openreview.net/pdf?id=uKZdlihDDn).

DGN and LDGN architectures can be seamlessly adapted to the **flow-matching training** framework, benefiting from faster sampling. Thus, we have added **Flow-Matching Graph Nets (FMGNs)** and **Latent FMGNs** to DGN4CFD. We have observed that these outperform their diffusion-based counterparts when the number of denoising steps is limited to 10 or fewer. However, for $\sim20$ or more denoising steps, diffusion models demonstrate superior performance (Appendix D.7).

Weights are also available and can be loaded as illustrated in these notebooks: [Ellipse](https://github.com/tum-pbs/dgn4cfd/blob/main/examples/Ellipse/inference.ipynb), [EllipseFlow](https://github.com/tum-pbs/dgn4cfd/blob/main/examples/EllipseFlow/inference.ipynb) and [Wing](https://github.com/tum-pbs/dgn4cfd/blob/main/examples/Wing/inference.ipynb).

## Datasets

All the datasets from our [paper](https://openreview.net/pdf?id=uKZdlihDDn) can be downloaded directly within python using our `DatasetDownloader`:
```python
import dgn4cfd as dgn

downloader = dgn.datasets.DatasetDownloader(
    dataset_url = dgn.datasets.DatasetUrl.<DATASET NAME>,
    path        = <DOWNLOAD PATH>,
)

dataset = dgn.datasets.pOnEllipse(
    path       = downloader.file_path,
    T          = <LENGTH OF SIMULATIONS>,
    transform  = <PRE-PROCESSING TRANSFORMATIONS>,
)

graph = dataset[<SIMULATION IDX>]
```

The datasets (`<DATASET NAME>`) available are (details on Appendix C):

- **pOnEllipse task**: Infer pressure on the surface of an ellipse immersed on a laminar flow ($Re \in [500, 1000]$ in the training dataset). Each simulation has 101 time-steps (`<LENGTH OF SIMULATIONS>`= 101). The training and testing datasets are:
  - pOnEllipseTrain (Training dataset)
  - pOnEllipseInDist
  - pOnEllipseLowRe
  - pOnEllipseHighRe
  - pOnEllipseThin
  - pOnEllipseThick

- **uvpAroundEllipse task**: Infer the velocity and pressure fields around an ellipse immersed on a laminar flow ($Re \in [500, 1000]$ in the training dataset).  Each simulation has 101 time-steps (`<LENGTH OF SIMULATIONS>`= 101). The training and testing datasets are:
  - uvpAroundEllipseTrain (Training dataset, 30.1 GB)
  - uvpAroundEllipseInDist
  - uvpAroundEllipseLowRe
  - uvpAroundEllipseHighRe
  - uvpAroundEllipseThin
  - uvpAroundEllipseThick

- **pOnWing task**: Infer pressure on the surface of a wing in 3D turbulent flow. The wing cross section is NACA 24XX airfoil. The geometry of the wings varies in terms of relative thickness, taper ratio, sweep angle, and twist angle. The training simulations have 251 time-steps each (`<LENGTH OF SIMULATIONS>`= 251), and the test simulations have 2501 time-steps each (`<LENGTH OF SIMULATIONS>`= 2501). The training and testing datasets are:
  - pOnWingTrain (Training dataset, 6.52 GB)
  - pOnWingInDist

## Examples
We recommend having a look at the training scripts and jupyter notebooks on the [examples/](https://github.com/tum-pbs/dgn4cfd/blob/main/examples) folder.
