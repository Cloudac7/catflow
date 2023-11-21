_default_input_dict = {
    "GLOBAL": {
        "PROJECT": "pmf",
        "RUN_TYPE": "MD"
    },
    "FORCE_EVAL": {
        "METHOD": "FIST",
        "PRINT": {
            "FORCES": {
                "_": "ON",
                "EACH": {}
            }
        },
        "MM": {
            "FORCEFIELD": {
                "CHARGE": [],
                "NONBONDED": {
                    "DEEPMD": []
                },
                "IGNORE_MISSING_CRITICAL_PARAMS": True
            },
            "POISSON": {
                "EWALD": {
                    "EWALD_TYPE": "none"
                }
            }
        },
        "SUBSYS": {
            "COLVAR": {
                "DISTANCE": {
                    "ATOMS": None
                }
            },
            "CELL": {
                "ABC": None
            },
            "TOPOLOGY": {}
        }
    },
    "MOTION": {
        "CONSTRAINT": {
            "COLLECTIVE": {
                "TARGET": None,
                "INTERMOLECULAR": True,
                "COLVAR": 1
            },
            "LAGRANGE_MULTIPLIERS": {
                "_": "ON",
                "COMMON_ITERATION_LEVELS": 20000000
            }
        },
        "MD": {
            "ENSEMBLE": "NVT",
            "STEPS": 20000000, # 10 ns
            "TIMESTEP": 0.5, # 0.5 fs
            "TEMPERATURE": None,
            "THERMOSTAT": {
                "CSVR": {
                    "TIMECON": 100,
                }
            }
        },
        "PRINT": {
            "TRAJECTORY": {
                "EACH": {
                    "MD": 1
                }
            },
            "FORCES": {},
            "RESTART_HISTORY": {
                "EACH": {
                    "MD": 200000
                }
            }
        }
    }
}