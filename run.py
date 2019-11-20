#!/usr/bin/env python3

import json

from azureml.core import Experiment, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import MpiConfiguration
from azureml.train.estimator import Estimator

# Load tenant ID from config.json; the tenant ID must be manually obtained 
# from the Azure portal:
config = json.load(open('config.json', 'rt'))
tenant_id = config['tenant_id']
interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id)

# This will open a web page to enable one to authenticate:
ws = Workspace.from_config(auth=interactive_auth)
mpi_conf = MpiConfiguration()

# Use local development environment:
compute_name = config['compute_name']
if compute_name == 'local':
    compute_target = compute_name

# Use AzureML compute target:
else:

    # Create compute target if it doesn't already exist:
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                               min_nodes=0,
                                                               max_nodes=6)
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

exp = Experiment(workspace=ws, name=config['experiment_name'])

cv = 3
to_run = Estimator(source_directory='.',
                   compute_target=compute_target,
                   entry_script='train.py',
                   script_params={'--cv': cv},
                   node_count=cv+2, # dask-mpi uses 2 nodes for its scheduler and client
                   conda_dependencies_file='env.yml',
                   distributed_training=mpi_conf)

run = exp.submit(to_run)
run.wait_for_completion(show_output=True)
