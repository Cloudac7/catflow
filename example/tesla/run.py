from miko_tasker.workflow.tesla import ClusterReactionWorkflow

task = ClusterReactionWorkflow(
    param_file="param-pmf.json",
    machine_pool="machine-cpu_lmp_vasp.json",
    conf_file="workflow_settings.yml"
)

task.run_loop()
