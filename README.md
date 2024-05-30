# Transfer and Meta-Learning Approaches to Passive Sonar Classification of Vessels in the Context of Few-shot Learning - Honours Project
This repository includes all the necessary work needed to acquire the results seen in my honours thesis. Various tools are included throughout this repository for preprocessing the dataset, defining models, training, graphing, testing and defining an SQL database for all the test results. This work is a fork of Lucas Domingos’ research as this honours project continues his work. The following document will describe the requirements of this project and how to run the various tools.


## Python Requirements
This project is based on Python3.8. A Dockerfile is included for development, although I did not use it for the honours project.

Install project Python dependencies with:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used for the this project is aquired from Lucas' IEEE Dataport page: https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data

The dataset includes one second captures of various vessels separated into three sets of inclusion and exclusion zones.

The dataset can be preprocessing using the `dataset_generator.py`. The paths have been hard coded into the script so will need to be changed when running the tool.

```bash
python3 src/dataset_generator.py
```

## Test Database
This project utilises a PostgreSQL database to store all training and testing data. The database definition are located in `database/` with an included Dockerfile and Docker compose file for running the PostgreSQL database. Using `docker-compose up` in the database directory will start the database, an external PostgreSQL client will need to be used to execute the database definition file.

## Training
The training of this project is achieved by using the `train.py` tool in the source folder. Necessary configuration files are located in `config_files/` for each test executed for this honours project. Before beginning training the Git submodule for required MAML libraries defined inside the source folder need to be initialised.

```bash
git submodule update --init --recursive
```
```bash
python3 src/train.py -c <path to confile file>
```

## Graphing
All graphing for this project was achieved through trial and error. To create the same graphs seen in the honours thesis, run the `graphing.ipynb` Python notebook with the database running.