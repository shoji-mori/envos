# ***envos***

## 1. Introduction
**Env**elope **O**bservation **S**imulator, developed by Shoji Mori.
This code executes synthetic observation by calculating physical model of young circumstellar systems (i.e. envelope and disk).

I really welcome improvements and requests from users. 

## 2. Features

1. **Flexible Model Generation**: `envos` enables users to generate models based on built-in models or numerical simulation data. The built-in models are available for each of the three regions, where the chosen models are combined into a physical model:
   - **Inner Envelope**: The Ulrich-Cassen-Moosman (UCM) model, a solution for the collapse of a rotating isothermal cloud core, based on the ballistic model of Ulrich (1976).
   - **Outer Envelope**: The Terebey-Shu-Cassen (TSC) model, which represents a self-similar solution of an isothermal sphere collapsing under its own gravity.
   - **Disk**: A power-law plus exponential-tail model, capable of representing a protoplanetary disk with specified mass, outer radius, scale height, and temperature.

2. **Consistent Temperature Structure**: `envos` calculates the temperature structure in a manner consistent with the provided density structure by using RADMC-3D (Dullemond et al. 2012; website: [https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/); github: [https://github.com/dullemond/radmc3d-2.0](https://github.com/dullemond/radmc3d-2.0)).

3. **Model Analysis**: `envos` offers tools for model analysis, in addition to usual physical structures, including:
   - **Column Density**: `envos` can calculate the column density in various directions (r-, θ-, z-direction, for now). Column density, a measure of the amount of material along a line of sight, is useful to compute optical depths.
   - **Streamlines**: `envos` can generate physical quantities along streamlines and visualize of the sreamlines in the model.

4. **Observational Simulation**: `envos` provides user-friendly tools for performing observational simulations via the RADMC-3D ray-tracing mode, which produces cube data (2 axes of a observational view and 1 axis of a line-of-sight velocity) in general, accounting for the effects of telescope beam size and finte resolution in the velocity. 

5. **Observation Data Analysis**: `envos` provides ways to analyse cube data obtained from simulated observations.
   - **Data Stacking and Slicing**: `envos` can generate integral intensity data, position-velocity data, and line profile data.
   - **Data Correlation**: `envos` can calculate data correlation between two observational data, a measure of the similarity between two data sets. This is particularly useful when comparing simulated and actual observations.

6. **Easy Configuration Management**: `envos` employs a user-friendly configuration system for easy management of parameters, enabling users to conveniently set up and modify their simulations.

7. **Multi-Core Processing**: `envos` is designed to leverage multi-core processing using OpenMP, allowing for efficient computation by parallelizing tasks and distributing them across multiple cores. This feature significantly accelerates the simulation and analysis processes, especially for large and complex models.

8. **Plotting Tools**: `envos` provides a comprehensive set of plotting tools for visualizing models and observation data. These tools can plot gas temperature, density, and velocity profiles, as well as observational outputs such as images and spectral line profiles.

9. **Python 3 Support**: All source codes in `envos` are written in Python 3 (version 3.6 or later), ensuring accessibility to a wide range of users and compatibility with modern Python environments.





<!--

- Density and velocity structures are calculated by a balistic model of Ulrich (1976), or one can input own kinematic data.  
- Temperature structure is calculated consistently with the density structure given by user.
- Calculation of temperature structure and sysnthetic observation is done by RADMC-3D (Dullemond et al. 2012; website: [https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/); github: [https://github.com/dullemond/radmc3d-2.0](https://github.com/dullemond/radmc3d-2.0)), which is commonly used in astronomical studies.
- All source codes are written in Python3 (ver. >= 3.6). 
-->

## 3. Requirements
- Python packages
    - numpy
    - scipy
    - dataclasses
    - pandas
    - astropy
    - matplotlib (for using RADMC-3D)
    - scikit-image

- To use RADMC-3D
     - Fortran compiler (e.g. `gfortan`, `intel fortran`)
     - (optional) Fortran openMP library

## 4. Setup
### 4.1 Install *RADMC-3D* and *radmc3dPy*
1. Download the installing source code from [github](https://github.com/dullemond/radmc3d-2.0):
`git clone https://github.com/dullemond/radmc3d-2.0.git`


2. Install RADMC-3D following the [HTML manual](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html) or [PDF version](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/radmc3d.pdf).
    1. `cd radmc3d-2.0/src`
    2. Edit Makefile if you need
    3. Generating an executable file `radmc3d`
       `make`
    5. Distribute `radmc3d` and radmc_tools in `$HOME/bin` in the default
       `make install`
    7. Add the path `$HOME/bin` to your \*rc file (e.g., ~/.bashrc)
       e.g. `echo export PATH=$HOME/bin:$PATH >> ~/.bashrc ` in command line
    6. Reload the \*rc file
       `source ~/.bashrc`

3. Check if `radmc3d` works following the RADMC-3D manual, executing example runs
    1. Move to `radmc3d-2.0/examples/run_simple_1`
    2. `python problem_setup.py`
    4. `radmc3d mctherm`
    5. Also `radmc3d mctherm setthreads 2` to check multithreading

4. Install `radmc3dPy` also following the manual.
    1. Move to `radmc3d-2.0/python/radmc3dPy`
    2. Execute setup.py: `python setup.py install --user`
       \*\* This `python` should also be python3. If you use python2, `radmc3dPy` will be installed into python2 libraries.

5. Check if `radmc3dPy` works (e.g. execute `python -c "import radmc3dPy"` in command line)


### 4.2 Install *envos*
1. Download `envos` from this github repository to your preferred location: 
`git clone https://github.com/mr91i/envos.git`

2. Check if it works e.g., execute an example script `python3 example_run.py`.


<!--

    * Put the dust opacity table and molecular line table that you want to use in RADMC-3D, into a directory.  Initially, (e.g., `storage/dustkappa_MRN20.inp`, `storage/molecule_c18o.inp`)dustkappa_XXX.inp and molecule_XXX.inp file can be found in directories of RADMC-3D package. One can also get any molecule_XXX.inp from [*Leiden Atomic and Molecular Database*](https://home.strw.leidenuniv.nl/~moldata/))

-->

## 5. Tutorial
Read and run `example_run.py`.


## 6. Parameters
Easiest way to controll envos is to use a class `envos.Config`.
In `envos.Config`, basic parameters user uses in envos are set.
Users create the instance of the `envos.Config` class setting below parameters and pass it to envos's classes.

Exapmple -- < *parameter name* >: < *type of parameter* > = < *default value* > 

### 6.1 General input
- `run_dir` : *str* = "./"  
The location of the executing directory. This directory is made automatically if not exists.  
- `fig_dir` : *str* = "./run/fig"  
The location of the directory in which the produced figures are saved. If not set, fig_dir is located in run_dir.
- `n_thread` : *int* = 1  
Number of threads used in RADMC-3D calculation. If this is > 2 and OpenMP is available, OpenMp is used for the radiative transfer calculations for thermal structure and line observation. Default is 1. 

### 6.2 Grid parameters
To generate a computational grid, one needs to set r, θ, φ coordinates as a list (1), or to set some parameters on grid (2).  
(1). Directly set axes
- `ri_ax`, `ti_ax`, `pi_ax` : *ndarary* or *list*  
Coordinates of the cell interface in r, θ, φ directions. When these parameters are set, below parameters are neglected.

(2). Generate axes inputting parameters
- `rau_in`, `rau_out` : *float*  
Coordinate of the cell interface of the inner and outer boundary in r direction, in au.  
- `theta_in`, `theta_out` : *float*  
Coordinate of the cell interface of the inner and outer boundary in θ direction, in radian. In default, this is from 0 to pi/2.  
- `phi_in`, `phi_out` : *float* 
Coordinate of the cell interface of the inner and outer boundary in φ direction, in radian. In default, this is from 0 to 2pi.  
- `nr`: *int*  
Number of cell in r direction.
- `ntheta`: *int*  
Number of cell in θ direction.
- `nphi`: *int*    
Number of cell in φ direction. 　
- `logr`: *bool* = True  
Set the r coordinate to be a logarithmic scale.
- `dr_to_r`: *float*  = None 
When logr is True and this parameter is set, the fraction of the cell length in the r direction to r is set to be `dr_to_r`. For example, when `dr_to_r` is 0.1, the cell size in the r-direction at 100 au is 1 au.
- `aspect_ratio`:  *float* = None
Ratio of the cell size in the θ or φ direction to dr.


### 6.3 Model parameters
Basically the kinetic structure of the UCM model is basically given by three parameters (see below for the definitions): `Mdot_smpy`, `Ms_Msun`, and `Omega`. However, `Mdot_smpy` can be given by `T`, `Ms_Msun` can be given by `t_yr`, and `Omega` can be given by `CR_au` or `jmid`.
- {`Mdot_smpy`, `T`}, {`Ms_Msun`, `t_yr`}, {`Omega`, `CR_au`, `jmid`} : *float*  
    - `Mdot_smpy`\[Msun/yr\] -- accretion rate
    - `T`\[K\] -- temperature of the molecular cloud core
    - `Ms_Msun` \[Msun\] -- central stellar mass
    - `t_yr`\[yr\] -- time from begining of the collapse 
    - `Omega`\[s^-1\] -- angular velocity of the parental cloud core   
    - `CR_au`\[au\] -- the centrifugal radius
    - `jmid`\[cm^2 s^-1\] -- specific angular momentum of the equatorial plane
    
- `meanmolw`: *float* = 2.3  
Mean molecular weight.
- `cavangle_deg`: *float* = 0.0  
Polar angle \[deg\] within which the density is deprecated, mimicking outflow cavities.
- `inenv`: *str* or *InnerEnvelope* = "UCM"  
The model used for the inner region of the envelope 
- `outenv`: *str* or *OuterEnvelope* = None  
The model used for the outer region of the envelope 
- `disk`: *str* or *Disk* = None  
The model used for the disk region 
- `rot_ccw`
If True, the rotation velocity becomes in the opposite direction

### 6.4 RADMC-3D parameters

- `nphot`: *int* = 1e6  
Number of photons used for thermal calculation

- `f_dg`: *float* = 0.01 
Dust-to-gas mass ratio

- `opac`: *str* 
Name of the opacity table, in the form of `storage/dustkappa_xxx.inp`.

- `Lstar_Lsun`: *float* = 1.0
The luminosity of the star, in units of solar luminosities.

- `mfrac_H2`: *float* = 0.74  
The mass fraction of H2 molecules in the gas. [Asplund et al. (2021)](https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract) would be helpful to determine this value.

- `molname`: *str*  
The name of the molecular line table to be used, in the form of `storage/molecule_xxx.inp`.

- `molabun`: *float*  
The abundance of the molecule of interest relative to H2.

- `iline`: *int*  
The line number in the table, starting from 1.

- `scattering_mode_max`: *int*   
A value of 1 or higher indicates that scattering is included. Refer to the RADMC3D manual for more details.

- `mc_scat_maxtauabs`: *int*  
The maximum optical depth of the photons that are scattered. Refer to the RADMC3D manual for more details.

- `tgas_eq_tdust`: *int* or *bool*   
Whether to set the gas temperature and dust temperature to be the same. The default is `True`. Setting it to `False` is not currently supported.

## 6.5 Observation parameters

- `dpc`: *float*  
Distance from the observer to the object, in pc.

- `size_au`: *float*  
Length of one side of the square observation area, in au.
If `size_au` is given, `sizex_au` and `sizey_au` will be ignored.

- `sizex_au`: *float*  
The vertical extent of the observation area, in au.

- `sizey_au`: *float*  
The horizontal extent of the observation area, in au.

- `pixsize_au`: *float*  
Pixel size, in astronomical units.

- `vfw_kms`: *float*  
Total velocity width, in kilometers per second.

- `dv_kms`: *float*  
Velocity resolution, in kilometers per second.

- `convmode`: *str*  
    - "normal": a standard convolution function in *astropy*
    - "fft": a convolution function in *astropy*, using fast fourier transform. This is faster than `normal` but requires more memory.
    - "scipy": a convolution function in *scipy*, slightly faster and less memory-intensive than `fft`.

- `beam_maj_au`: *float*  
The major axis of the beam, in astronomical units.

- `beam_min_au`: *float*  
The minor axis of the beam, in astronomical units.

- `vreso_kms`: *float*  
Full velocity width of convolution, in kilometers per second.

- `beam_pa_deg`: *float*  
The position angle of the beam, in degrees.

- `incl`: *float*  
Angle, in degrees, between the polar axis and the line of sight (`incl`=0: face-on, `incl`=90: edge-on)

- `phi`: *float*  
The longitude, in degrees.

- `posang`: *float*  
The position angle of the camera, in degrees.


<!--
## 7. Functions/Classes
-->

## 7. Input Files

`envos` requires the input files to execute RADMC-3D: dust opacity and molecular line.
The input files are expected to be located in `storage` directory. We here produce some exapmples 
of the input files for RADMC-3D, in order to skip the time-consuming process of gathering the input
files. However, if you use these files, please do not forget to cite the original papers.

### 7.1 Dust Opacity:
dustkappa_silicate.inp is taken from the default opacity table used in RADMC-3D. 
This is for Amorphous Olivine with 50% Mg and 50% Fe
Please do not forget to cite in your publications the original paper of these optical constant measurements:
*Jaeger, Mutschke, Begemann, Dorschner, Henning (1994) A&A 292, 641-655;*
*Dorschner, Begemann, Henning, Jaeger, Mutschke (1995) A&A 300, 503-520.*
File was made with the makedustopac.py code by Cornelis Dullemond using the bhmie.py Mie code of *Bohren and 
Huffman* (python version by *Cornelis Dullemond*, from original bhmie.f code by *Bruce Draine*)
Prameter: Grain size =  1.000000e-05 cm; Material density =  3.710 g/cm^3

dustkapp_MRN20.inp is calculated by dsharp_opac (https://github.com/birnstiel/dsharp_opac), 
which is developed by *Tilman Birnstiel*. The ice fraction is 20 wt%; the dust size distribution follows,  
*Mathis, Rumpl, Nordsieck, ApJ, Vol. 217, p. 425-433 (1977)*. 
When you use the table, please cite: 
*Birnstiel, T., Dullemond, C. P., Zhu, Z., et al. 2018, ApJL, 869, L45*.
In addition, please do not forget to cite in your publications the original paper of these optical constant measurements:
*Henning, T., & Stognienko, R. 1996, A&A, 311, 291;*
*Draine, B. T. 2003, ApJ, 598, 1017;*
*Warren, S. G., & Brandt, R. E. 2008, Journal of Geophysical Research (Atmospheres), 113, D14220.*
Plese see *Birnstiel et al. 2018* for the detail.

### 7.2 Molecular line:
molecule_c18o.inp and molecule_c3h2.inp are taken from *LAMDA* (Leiden Atomic and Molecular Database; 
https://home.strw.leidenuniv.nl/~moldata/). When you use this data and the database, please follow the citation 
rule decribed in LAMDA webpage.
