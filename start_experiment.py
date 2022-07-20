from azureml.core import (
    Workspace,
    Experiment,
    Environment,
    RunConfiguration,
    ScriptRunConfig,
)


#Defines compute instance
compute_name = 'hackcompute22'

#Defines the name of experiment and environment
experiment_name = 'Hack-experiment'
environment_name = 'Hack-environment'

#Defines name of model and where it is saved
model_name = 'model.pkl'
model_path = 'outputs/model.pkl'

#Defines  directory to run the experiment from
source_directory = '.'

#Defines the entry script for the experiment
script_path = 'Hack Partners/traffic.py'

#Defines the location of the machine learning workspace
subscription_id = '4f67948b-2ff9-49ee-bf1f-90c32dc7545e'
resource_group  = 'cloud-shell-storage-westeurope'
workspace_name  = 'Hack_Workspace'

#Connect to workspace
ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)

#Uses workspace to create an experiment
exp = Experiment(workspace=ws, name=experiment_name)

#Creates environment with the packages needed
env = Environment(name=environment_name)

for pip_package in ['numpy','pandas','install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html', 'torch torchvision torchaudio', 'matplotlib']:
    env.python.conda_dependencies.add_pip_package(pip_package)

#Creates a run configuration to connect environment and compute
run_config = RunConfiguration()
run_config.target = compute_name
run_config.environment = env


#Creates a script run config to tie all the elements together
config = ScriptRunConfig(
    source_directory=source_directory, script=script_path, run_config=run_config
)

# submitting the experiment to start it
run = exp.submit(config)

#waits for completion and shows output of the experiment
run.wait_for_completion(show_output=True, wait_post_processing=True)

#Registers the model with the experiment
run.register_model(model_name=model_name, model_path=model_path)
