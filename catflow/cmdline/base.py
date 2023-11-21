import fire
from .calculation.vasp import vasprun, trajcheck
from .calculation.dpgen import simu
from .workflow.pmf import pmf
from .workflow.tesla import tesla, tesla_cluster, tesla_metad


tasker = {
    'vasprun': vasprun,
    'trajcheck': trajcheck,
    'simu': simu,
    'pmf': pmf,
    'tesla': tesla,
    'tesla_cluster': tesla_cluster,
    'tesla_metad': tesla_metad
}


def hello(name):
    return 'Hello {name}!'.format(name=name)


def main():
    fire.Fire(
        {
            'tasker': tasker,
            'hello': hello
        }
    )
