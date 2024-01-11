# Computer Vision III: Detection, Segmentation, and Tracking (IN2375)	
# Technical University Munich - WS 2023/24

## 1. Python Setup

If you are doing these exercises in google colab, you can skip the next section and directly go to the notebook `0_exercise_intro.iypnb` in the folder `exercise_00`.

For the following description, we assume that you are using Linux or MacOS and that you are familiar with working from a terminal. The exercises are implemented in Python 3.10, so this is what we are going to install here.

If you are using Windows, you will have to google or check out the forums for setup help from other students. There are plenty of instructions for Anaconda for Windows using a graphical user interface though.

To avoid issues with different versions of Python and Python packages we recommend that you always set up a project-specific virtual environment. The most common tools for clean management of Python environments are *pyenv*, *virtualenv* and *Anaconda*. For simplicity, we are going to focus on Anaconda.

### Anaconda setup (Locally)
Download and install miniconda (minimal setup with less start up libraries) or conda (full install but larger file size) from [here](https://www.anaconda.com/products/distribution#Downloads). Create an environment using the terminal command:

`conda env create -f environment.yaml`

Next, activate the environment using the command:

`conda activate cv3dst`

Start a jupyter notebook using:

`jupyter notebook`

Jupyter notebooks use the python version of the current active environment so make sure to always activate the `cv3dst` environment before working on notebooks for this class. You can now go to the notebook `0_exercise_intro.iypnb` in the folder `exercise_00`.

## 2. Exercise Download

The link for downloading the exercises will be provided in the exercise slides and on the course [website](https://cvg.cit.tum.de/teaching/ws2023/cv3)/moodle. Each time we start a new exercise you will have to unzip the exercise and copy it into the current directory as we are utilizing some shared folders.
### The directory layout for the exercises

    cv3dst
    ├── datasets            # The datasets will be stored here
    ├── exercise_00                 
    ├── exercise_01                 
    ├── exercise_02                     
    ├── exercise_03                    
    ├── exercise_04
    ├── exercise_05
    ├── models              # Where you will find all trained models, not for uploading
    ├── output              # Where you will find zipped exercises for uploading
    ├── README.md
    ├── requirements.txt    # This contains the addtional package needed in google colab
    └── environment.yaml    # This contains the full python environment


## 3. Dataset Download

Datasets will generally be downloaded automatically by exercise notebooks and stored in a common datasets directory shared among all exercises. A sample directory structure for cifar10 dataset is shown below:-

    cv3dst
        └── datasets                    # The datasets required for all exercises will be downloaded here
            ├── FashionMNIST            # Dataset directory
            │   └── ...                 # Dataset files 
            ├── ...                     # More datasets
            └── ...                     # More datasets


## 4. Exercise Submission
Your trained models will be automatically evaluated on a test set on our server. To this end, login or register for an account at:

[https://cv3dst.cvai.cit.tum.de/login](https://cv3dst.cvai.cit.tum.de/login)

Note that only students who have registered for this class in TUM Online can register for an account. By using your matriculation number we send login data to your associated email address (probably the tum provided account if you didn't change it).

After you have worked through an exercise, execute the notebook cells that save and zip the exercise. The output can be found in the global `cv3dst/output` folder.

You can login to the above website and upload your zip submission for the current exercise. Your submission will be evaluated by our system.

You will receive an email notification with the results upon completion of the evaluation. To make the exercises more fun, you will be able to see a leaderboard of everyone's (anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline of the current exercise. Whereas the email contains the result of the current evaluation, the entry in the leaderboard always represents the best score for the respective exercise.

## 5. Acknowledgments

We thank the **TU Munich Visual Computing and Artificial Intelligence Group** for creating this exercise framework.
