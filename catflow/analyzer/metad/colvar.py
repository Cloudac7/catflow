from pathlib import Path

import numpy as np
import pandas as pd

class Colvar:
    def __init__(self, 
                 file: str, 
                 bias_name: str) -> None:
        colvar_file_path = Path(file)
        self.colvar_key = []
        self._parse_colvar_file(colvar_file_path)
        self.bias_name = bias_name
        for key in self.colvar_key:
            if bias_name + '.bias' in key:
                self.bias_key = key
                self.bias = self.colvars[self.bias_key].to_numpy(dtype=np.float64)

    def _parse_colvar_file(self, file: Path) -> None:
        with open(file, 'r') as f:
            first_line = f.readline()
            if first_line.startswith("#! FIELDS"):
                self.colvar_key = self._parse_header(first_line)
        self.colvars = pd.read_csv(file, delim_whitespace=True, comment="#", names=self.colvar_key)
        
    def _parse_header(self, line: str) -> None:
        """
        #! FIELDS time dis_oo.min dis_GO.max d_oo s_oo.max cn_ago.max dis_GO2.min opes.bias opes.rct opes.zed opes.neff opes.nker
        """
        columns = line.split()
        return columns[2:] # type: ignore
