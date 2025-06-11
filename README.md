# Gamma vs Hadron Classification
This project aims to classify data from the MAGIC Gamma Telescope using different approaches, including standard machine learning methods (specifically XGBoost), as well as deep learning models such as Multilayer Perceptrons and Convolutional Neural Networks.  
The dataset analyzed was sourced from Kaggle and can be accessed at: https://www.kaggle.com/datasets/abhinand05/magic-gamma-telescope-dataset  

## MAGIC Telescope
MAGIC (Major Atmospheric Gamma Imaging Cherenkov telescope), located on the island of La Palma, is currently one of the largest ground-based gamma ray telescope.  
It detects particle showers produced by very high-energy gamma rays (on the order of TeV), thanks to the use of the Imaging Air Cherenkov Technique (IACT). When gamma rays interact electromagnetically with the Earth's atmosphere, they generate secondary particle which in turn yield a new generation of $\gamma$ rays through bremsstrahlung. Any secondary particle traveling faster than the speed of light in the atmosphere emits Cherenkov radiation, which is captured by the telescope. Throght the detection of this light images, it's possible to reconstruct both the longitudinal and lateral developmente of the electromagnetic shower, as well as determine the arrival direction of the primary gamma ray.  

<table>
  <tr>
    <td><img src="images/magic.jpeg" alt="MAGIC Telescope" width="500"/></td>
    <td><img src="images/iact.png" alt="IACT Method" width="300"/></td>
  </tr>
</table>

## Gamma versus Hadron discrimination
Many gamma ray experiments have to deal with the problem of separating showers produced by interesting gammas signals from the vast hadrons events background.
The main differences between the two are the following:
- **Electromagnetic showers**: They can be initiated by photons and develop due combination of processes like pair production and bremsstrahlung.

- **Hadronic showers**: These have more complex descriptions as they involve electromagnetic but also strong interactions, giving rise to different components. It can be divided into an hadronic component and an eletromagnetic one that arise thank to the production of neutral pions which in turn will generate gamma particles.

<p align="center">
  <img src="images/gamma_hadrons.png" alt="Comparison between gamma and hadron signal" width="500"/>
</p>

The task of separating both types of particles is an occuring problem in the area of ground-based $\gamma$-ray astronomy and is commonly referred to as gamma/hadron separation.  
To do this the recorded image, after a necessary pre-processing, is parameterized into `Hillas' parameters, mainly a set of second moments which include image shape parameters (length (L) and width (W)) and image orientation parameters, like azwidth (A) and alpha ($\alpha$). Both simulation and experimental studies have shown that γ-ray images are more regular and compact with smaller L and W as compared with their cosmic-ray counterparts and have a well-defined major axis (orientation) which, in the case of γ-rays coming from a point γ-ray source, are oriented closer towards the telescope axis.  

