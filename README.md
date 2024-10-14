# lens-npe

This repository contains source code for Erickson et al. 2024 "Lens Modeling of 
STRIDES Strongly Lensed Quasars using Neural Posterior Estimation"

To reproduce all figures in the paper, use Paper/make_figures.ipynb. This will
require a ([Zenodo download](https://zenodo.org/records/13906030)). To recreate
figures, you do NOT need to download trained_models.tgz (26 GB!).

This code base heavily relies on the use of paltas (https://github.com/swagnercarena/paltas). Note that while updates 
from this project are being merged into the paltas main branch, the @smericks fork of 
paltas contains all necessary updates: 
    https://github.com/smericks/paltas

See Notebooks for demonstrations of key analysis components:
- configurations for training neural networks (training_networks.ipynb)
- feeding images to a neural network to generate mass model posteriors (make_predictions.ipynb)
- hierarchical Bayesian inference for lens mass population models (hierarchical_inference.ipynb)
- hierarchical re-weighting of individual posteriors to account for out-of-distribution shift (posterior_reweighting.ipynb)

Note: some notebooks also require the Zenodo download


