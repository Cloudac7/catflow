import asyncio

from miko_tasker.utils.config import load_yaml_configs
from miko_tasker.flows.pmf_flow import PMFInput, PMFOutput
from miko_tasker.flows.pmf_flow import flow_pmf_calculation

if __name__ == '__main__':
    flow_input = PMFInput.parse_obj(load_yaml_configs("config.yaml"))
    flow_output = PMFOutput(pmf_tasks=[])
    asyncio.run(flow_pmf_calculation(
        flow_input=flow_input,
        flow_output=flow_output
    ))

