import sys
from pathlib import Path
from dynaconf import Dynaconf


settings = Dynaconf(
    __name__,
    envvar_prefix='MIKO',
    settings_files=[
        'machine.yml', 'machine.local.yml'
    ],
    environments=False,
    load_dotenv=True,
    ENVVAR_FOR_DYNACONF='MIKO_MACHINE',
    includes=[
        Path(sys.prefix, 'etc', 'miko', 'machine.yml'),
    ]
)
