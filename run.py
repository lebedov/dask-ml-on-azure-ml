#!/usr/bin/env python3

import json

from azureml.core import Experiment, Workspace, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import MpiConfiguration, RunConfiguration, DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE
from azureml.train.estimator import Estimator

# Load tenant ID from config.json; the tenant ID must be manually obtained 
# from the Azure portal:
config = json.load(open('config.json', 'rt'))
tenant_id = config['tenant_id']
interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id)

# This will open a web page to enable one to authenticate:
ws = Workspace.from_config(auth=interactive_auth)
run_conf = RunConfiguration()
run_conf.framework = 'Python'

# Number of folds for cross validation; if set to None, no cross validation is
# performed (and hence dask is not used):
cv = 3

# Use local development environment:
compute_name = config['compute_name']
if compute_name == 'local':
    run_conf.environment.python.user_managed_dependencies = True
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

    run_conf.target = compute_target
    run_conf.environment.docker.enabled = True
    run_conf.environment.docker.base_image = DEFAULT_CPU_IMAGE
    run_conf.environment.python.conda_dependencies = \
        CondaDependencies(conda_dependencies_file_path='env.yml')
    run_conf.environment.python.user_managed_dependencies = False
    if cv:
        run_conf.communicator = 'OpenMPI'
        run_conf.mpi = MpiConfiguration()
        run_conf.node_count = cv+2
exp = Experiment(workspace=ws, name=config['experiment_name'])

use_estimator = True
if use_estimator:
    if cv:
        script_params = {'--cv': cv}
        node_count = cv+2 # dask-mpi uses 2 nodes for its scheduler and client
        distributed_training = MpiConfiguration()
    else:
        script_params = None
        node_count = None
        distributed_training = None
    to_run = Estimator(source_directory='.',
                       compute_target=compute_target,
                       entry_script='train.py',
                       script_params=script_params,
                       node_count=node_count,
                       use_gpu=False,
                       conda_dependencies_file='env.yml',
                       distributed_training=distributed_training)
else:
    if cv:
        arguments = ['--cv', str(cv)]
    else:
        arguments = []
    to_run = ScriptRunConfig(source_directory='.',
                             script='train.py',
                             arguments=arguments,
                             run_config=run_conf)
run = exp.submit(to_run)
run.wait_for_completion(show_output=True)
