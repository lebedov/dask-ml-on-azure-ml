Using Dask-ML on Azure ML
=========================

This repository contains a simple demo of how to run 
[dask-ml](https://ml.dask.org/) functions
on an Azure ML compute cluster. The demo takes advantage of [dask-mpi](https://mpi.dask.org) to simplify cluster setup.

Instructions
------------

- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)  

- Create and activate a Python 3 environment:

        conda create azureml
        conda activate azureml

- Install Azure ML SDK:

        pip install azureml-sdk

- [Create](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace) a new Azure ML workspace

- Clone this repository and create a `config.json` file in the repository directory containing your Azure ML subscription, tenant ID, resource group, workspace name, and your preferred names for the compute cluster and experiment. The file should look like the following:

        {
            "tenant_id": "WWWWWWWW-WWWW-WWWW-WWWW-WWWWWWWWWWWW",
            "subscription_id":"XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX", 
            "resource_group": "YYYYYYYY",
            "workspace_name": "ZZZZZZZZ",
            "compute_name": "AAAAAAAA",
            "experiment_name": "BBBBBBBB"
        }

- Run the demo as follows:

        python run.py

- Once the demo has finished, you can view the results in the Azure portal.
