import os
import numpy as np
import pandas as pd

import pytest
from catflow.tasker.utils.config import load_yaml_configs
from catflow.tasker.flows.pmf_flow import SafeList
from catflow.tasker.flows.pmf_flow import PMFInput, PMFOutput, PMFTaskOutput
from catflow.tasker.flows.pmf_flow import melting_test_temperatures
from catflow.tasker.flows.pmf_flow import subflow_melting_zone_search
from catflow.tasker.flows.pmf_flow import subflow_pmf_temperature_range
from catflow.tasker.flows.pmf_flow import flow_pmf_calculation


def compare_list(actual, expected):
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_pmf_config(shared_datadir):
    config_dict = load_yaml_configs(shared_datadir / "config.yaml")
    config = PMFInput.parse_obj(config_dict)
    assert config.flow_config.coordinates == [1.4, 2.0, 3.8]
    assert config.flow_config.t_min == 300.
    assert config.job_config.work_path == "."
    assert config.job_config.command == "dummy_command"
    assert type(config.job_config.resources) == dict
    assert config.job_config.job_name == "PMF"
    #assert config.init_structure_path == "last_frame.xyz"
    assert config.job_config.reaction_pair == [0, 1]
    assert config.flow_config.cluster_component == ["Pt"]


@pytest.fixture
def pmf_input(shared_datadir):
    config_dict = load_yaml_configs(shared_datadir / "config.yaml")
    config = PMFInput.parse_obj(config_dict)
    return config


@pytest.fixture
def pmf_output():
    flow_output = PMFOutput(pmf_tasks=[])
    flow_output.pmf_tasks += [
        PMFTaskOutput(
            coordinate=1.4,
            temperature=300.,
            task_path=".",
            last_frame_path="last_frame.xyz",
            restart_time=0,
            convergence=True,
            lindemann_index=0.05
        ),
        PMFTaskOutput(
            coordinate=1.4,
            temperature=400.,
            task_path=".",
            last_frame_path="last_frame.xyz",
            restart_time=0,
            convergence=True,
            lindemann_index=0.05
        ),
        PMFTaskOutput(
            coordinate=1.4,
            temperature=600.,
            task_path=".",
            last_frame_path="last_frame.xyz",
            restart_time=0,
            convergence=True,
            lindemann_index=0.3
        ),
        PMFTaskOutput(
            coordinate=1.4,
            temperature=700.,
            task_path=".",
            last_frame_path="last_frame.xyz",
            restart_time=0,
            convergence=False
        ),
    ]
    return flow_output


@pytest.fixture
def flow_input_output(mocker, pmf_input, pmf_output):
    mocker.patch(
        "catflow.tasker.flows.pmf_flow.task_pmf_calculation",
        mock_task_pmf_calculation
    )
    mocker.patch(
        "catflow.tasker.flows.pmf_flow.lindemann_index_calculation",
        mock_task_lindemann_index_calculation
    )
    return pmf_input, pmf_output


def test_melting_test_temperatures():
    temps = np.array([400., 500., 600., 700, 800.])
    assert np.equal(melting_test_temperatures(), temps).all()

    t_min = 400
    t_max = 800
    step = 100.
    assert np.equal(melting_test_temperatures(
        t_min=t_min, t_step=step), temps).all()
    assert np.equal(melting_test_temperatures(
        t_max=t_max, t_step=step), temps).all()
    assert np.equal(
        melting_test_temperatures(
            t_min=t_min, t_max=t_max, t_step=step),
        temps).all()


async def mock_task_pmf_calculation(
    coordinate,
    temperature,
    flow_input: PMFInput,
    pmf_task_outputs: SafeList,
    init_structure_path=None,
    restart_time=0
):
    task_output = PMFTaskOutput(
        coordinate=coordinate,
        temperature=temperature,
        task_path=".",
        last_frame_path="last_frame.xyz",
        restart_time=0
    )
    await pmf_task_outputs.add_item(task_output)
    return task_output


def mock_task_lindemann_index_calculation(
    flow_input: PMFInput,
    task_output: PMFTaskOutput
):
    temperature = task_output.temperature
    if temperature >= 600.:
        return [0.3, 0.3, 0.3]
    elif temperature <= 500.:
        return [0.05, 0.05, 0.05]
    else:
        return [0.15, 0.15, 0.15]

@pytest.mark.parametrize(

    "temperatures, expected_task_length", [
        ([400., 500., 533., 567., 600., 700.], 7),
        ([300., 400., 500., 600., 700., 800., 900.], 9),
        ([500., 600., 700., 800., 900., 1000.], 10),
        ([200., 300., 400., 500., 600.], 8),
        ([200., 300., 400.], 9),
        ([600., 700., 800., 900.], 10)
    ]
)
@pytest.mark.asyncio
async def test_melting_temp_check(
    temperatures,
    expected_task_length,
    flow_input_output
):
    _ = flow_input_output
    flow_input = _[0]
    flow_output = _[1]

    pmf_task_outputs = SafeList()
    for task in flow_output.pmf_tasks:
        await pmf_task_outputs.add_item(task)

    indexes, temps = await subflow_melting_zone_search(
        coordinate=1.4,
        temperatures=temperatures,
        flow_input=flow_input,
        pmf_task_outputs=pmf_task_outputs
    )
    tasks = await pmf_task_outputs.get_items()
    assert len(tasks) == expected_task_length
    assert np.equal(temps, [400., 500., 533., 567., 600., 700.]).all()
    compare_list(indexes, [2, 3])


@pytest.mark.asyncio
async def test_melting_temp_check_extend_with_melting(flow_input_output):
    flow_input = flow_input_output[0]
    flow_output = flow_input_output[1]

    pmf_task_outputs = SafeList()
    for task in flow_output.pmf_tasks:
        await pmf_task_outputs.add_item(task)
    
    indexes, temps = await subflow_melting_zone_search(
        coordinate=1.4,
        temperatures=[525., 550., 575., 600.],
        flow_input=flow_input,
        pmf_task_outputs=pmf_task_outputs
    )
    tasks = await pmf_task_outputs.get_items()
    assert len(tasks) == 9
    compare_list(indexes, [2, 3, 4])
    assert np.equal(temps, [325., 425., 525., 550., 575., 600., 700.]).all()


@pytest.mark.asyncio
async def test_melting_temp_check_extend_not_melted(flow_input_output):
    flow_input = flow_input_output[0]
    flow_output = flow_input_output[1]

    pmf_task_outputs = SafeList()
    for task in flow_output.pmf_tasks:
        await pmf_task_outputs.add_item(task)

    flow_input.flow_config.t_step = 25.
    indexes, temps = await subflow_melting_zone_search(
        coordinate=1.4,
        temperatures=[400., 425., 450., 475., 500., 525., 550.],
        flow_input=flow_input,
        pmf_task_outputs=pmf_task_outputs
    )

    tasks = await pmf_task_outputs.get_items()
    assert len(tasks) == 12
    compare_list(indexes, [2, 3, 4])
    assert np.equal(temps, [475., 500., 525., 550., 575., 600., 625.]).all()


@pytest.mark.asyncio
async def test_melting_temp_check_extend_all_melting(flow_input_output):
    flow_input = flow_input_output[0]
    flow_output = flow_input_output[1]

    pmf_task_outputs = SafeList()
    for task in flow_output.pmf_tasks:
        await pmf_task_outputs.add_item(task)

    indexes, temps = await subflow_melting_zone_search(
        coordinate=1.4,
        temperatures=[525.0, 575.0],
        flow_input=flow_input,
        pmf_task_outputs=pmf_task_outputs
    )
    tasks = await pmf_task_outputs.get_items()
    assert len(tasks) == 10
    compare_list(indexes, [2, 3, ])
    assert np.equal(temps, [325., 425., 525., 575., 675., 775.]).all()


@pytest.mark.asyncio
async def test_melting_point_test_each_state(flow_input_output):
    flow_input = flow_input_output[0]
    flow_output = flow_input_output[1]

    pmf_task_outputs = SafeList()
    for task in flow_output.pmf_tasks:
        await pmf_task_outputs.add_item(task)

    indexes, temps = await subflow_melting_zone_search(
        coordinate=1.4,
        temperatures=[400., 500., 600., 700.],
        flow_input=flow_input,
        pmf_task_outputs=pmf_task_outputs
    )
    tasks = await pmf_task_outputs.get_items()
    assert len(tasks) == 7
    compare_list(indexes, [2, 3])
    assert np.equal(temps, [400., 500., 533., 567., 600., 700.]).all()


def mock_phase_transition_lindemann_index(
    flow_input: PMFInput,
    task_output: PMFTaskOutput
):
    coordinate = task_output.coordinate
    temperature = task_output.temperature
    # IS: 400-600K
    if coordinate <= 1.5:
        if temperature >= 600.:
            return [0.3, 0.3, 0.3]
        elif temperature <= 400.:
            return [0.05, 0.05, 0.05]
        else:
            return [0.15, 0.15, 0.15]
    # FS: 550-750K
    elif coordinate >= 3.5:
        if temperature >= 750.:
            return [0.3, 0.3, 0.3]
        elif temperature <= 550.:
            return [0.05, 0.05, 0.05]
        else:
            return [0.15, 0.15, 0.15]
    # TS: 450-650K
    else:
        if temperature >= 650.:
            return [0.3, 0.3, 0.3]
        elif temperature <= 450.:
            return [0.05, 0.05, 0.05]
        else:
            return [0.15, 0.15, 0.15]


@pytest.fixture
def pmf_flow_config(mocker, pmf_input, pmf_output):
    mocker.patch(
        "catflow.tasker.flows.pmf_flow.task_pmf_calculation",
        mock_task_pmf_calculation
    )
    mocker.patch(
        "catflow.tasker.flows.pmf_flow.lindemann_index_calculation",
        mock_phase_transition_lindemann_index
    )
    return pmf_input, pmf_output


@pytest.mark.asyncio
async def test_pmf_temperature_range(pmf_flow_config):

    flow_input = pmf_flow_config[0]
    flow_output = pmf_flow_config[1]
    pmf_task_outputs = SafeList()
    for task in flow_output.pmf_tasks:
        await pmf_task_outputs.add_item(task)
    
    flow_input.flow_config.t_min = 300.

    temps = await subflow_pmf_temperature_range(
        flow_input=flow_input,
        pmf_task_outputs=pmf_task_outputs
    )

    tasks = await pmf_task_outputs.get_items()
    assert len(tasks) == 12
    assert np.equal(temps, [300., 400., 500., 600., 700., 800., 900.]).all()


@pytest.mark.asyncio
async def test_pmf_temperature_range_2(pmf_flow_config):

    flow_input = pmf_flow_config[0]
    flow_output = pmf_flow_config[1]
    pmf_task_outputs = SafeList()
    for task in flow_output.pmf_tasks:
        await pmf_task_outputs.add_item(task)
    flow_input.flow_config.temperatures = [300., 900.]

    temps = await subflow_pmf_temperature_range(
        flow_input=flow_input,
        pmf_task_outputs=pmf_task_outputs
    )

    tasks = await pmf_task_outputs.get_items()
    assert len(tasks) == 14
    assert np.equal(temps, [200., 300., 500., 700., 900., 1000.]).all()


async def mock_convergence_test_lagrange_multiplier(
    task_output: PMFTaskOutput,
    pmf_task_outputs: SafeList
):
    import asyncio
    import random
    from catflow.tasker.utils.statistics import block_average

    np.random.seed(114514)
    fluctuation = (np.random.rand(40000) - 0.5) * 1e-3

    coordinate = task_output.coordinate
    dummy_pmf = {
        1.4: 0.0, 1.7: -3.0, 2.0: 0.0, 2.2: 1.0, 3.0: 0.5, 3.8: 0.0
    }
    lagrange_mults = np.ones(40000) * dummy_pmf[coordinate]
    lagrange_mults += fluctuation

    sleeptime = random.randint(0, 300)
    await asyncio.sleep(float(sleeptime/100))

    mean, var = block_average(
        lagrange_mults[1:], int(len(lagrange_mults[1:])/10)
    )
    if var < 0.1:
        task_output.convergence = True
        task_output.pmf_mean = mean
        task_output.pmf_var = var
    await pmf_task_outputs.add_item(task_output)
    return task_output


@pytest.mark.asyncio
async def test_pmf_calculation_workflow(mocker, pmf_flow_config, tmp_path):

    mocker.patch(
        "catflow.tasker.flows.pmf_flow.convergence_test_lagrange_multiplier",
        mock_convergence_test_lagrange_multiplier
    )
    coordinates = [1.4, 1.7, 2.0, 2.2, 3.0, 3.8]
    temps = [300., 400., 500., 600., 700., 800., 900.]

    flow_input = pmf_flow_config[0]
    flow_output = pmf_flow_config[1]
    flow_input.job_config.work_path = str(tmp_path)
    flow_input.flow_config.t_min = 300.
    flow_input.flow_config.coordinates = coordinates

    flow_output = await flow_pmf_calculation(
        flow_input=flow_input,
        flow_output=flow_output
    )

    assert len(flow_output.pmf_tasks) == 42

    df = pd.DataFrame(
        {
            "coordinate": [task.coordinate for task in flow_output.pmf_tasks],
            "temperature": [task.temperature for task in flow_output.pmf_tasks],
            "convergence": [task.convergence for task in flow_output.pmf_tasks],
            "pmf_mean": [task.pmf_mean for task in flow_output.pmf_tasks],
            "pmf_var": [task.pmf_var for task in flow_output.pmf_tasks]
        }
    )

    assert df.shape == (42, 5)
    assert set(df["coordinate"].unique().tolist()) == set(coordinates)
    assert all(df["convergence"] == True)

    saved_flow_output = load_yaml_configs(tmp_path / "pmf_flow_output.yaml")
    assert saved_flow_output["pmf_tasks"] == flow_output.pmf_tasks

    assert os.path.exists(tmp_path / "pmf.png")
