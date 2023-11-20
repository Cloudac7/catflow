import numpy as np
from typing import Optional, Tuple, List

class Hills:
    """
    Loads and extracts information from HILLS files.

    Args:
        name (str, optional): Name of the HILLS file. Defaults to "HILLS".
        encoding (str, optional): Encoding of the HILLS file. Defaults to "utf8".
        ignoretime (bool, optional): If True, the time values in the HILLS file are ignored. 
            If False, the time values are saved. Defaults to True.
        periodic (List[bool], optional): List of boolean values indicating which CV is periodic. 
            Defaults to [False, False].
        cv_per (List[List[float]], optional): List of lists containing two numeric values defining 
            the periodicity of each periodic CV. Defaults to [[-numpy.pi, numpy.pi]].
        timestep (float, optional): Time difference between hills, in picoseconds.

    Attributes:
        hills_filename (str): Name of the HILLS file.
        cvs (int): Number of collective variables (CVs) in the HILLS file.
        cv_name (List[str]): Names of the CVs.
        periodic (ndarray): Array of boolean values indicating which CV is periodic.
        cv_per (List[List[float]]): List of lists containing two numeric values defining 
            the periodicity of each periodic CV.
        cv (ndarray): Array of CV values.
        cv_min (ndarray): Array of minimum CV values.
        cv_max (ndarray): Array of maximum CV values.
        sigma (ndarray): Array of sigma values.
        heights (ndarray): Array of hill heights.
        biasf (ndarray): Array of bias factors.

    Raises:
        ValueError: If cv_per is not provided for each periodic CV.

    Examples:
        Load a HILLS file and extract its information:

        >>> hills = Hills(name="my_hills_file")
        >>> print(hills.get_cv_name())
        ['cv1', 'cv2']
        >>> print(hills.get_periodic())
        [False, True]
        >>> print(hills.get_cv_per())
        [[-3.141592653589793, 3.141592653589793], [-numpy.pi, numpy.pi]]

    """

    def __init__(
            self, 
            name="HILLS", 
            encoding="utf8", 
            ignore_time=True, 
            periodic=None, 
            cv_per: Optional[List[List[float]]] = None,
            time_step=None
    ):
        self.read(
            name, encoding, ignore_time, periodic, cv_per, time_step
        )
        self.hills_filename = name
    
    def read(
            self, 
            name="HILLS", 
            encoding="utf8", 
            ignore_time=True, 
            periodic=None, 
            cv_per: Optional[List[List[float]]] = None,
            time_step=None
    ):
        with open(name, 'r', encoding=encoding) as hills_file:
            first_line = hills_file.readline()
        columns = first_line.split() 
        number_of_columns_head = len(columns) - 2
        self.cvs = (number_of_columns_head - 3) // 2
        self.cv_name = columns[3:3+self.cvs]

        if periodic == None:
            periodic = [False for i in range(self.cvs)]
        self.periodic = np.array(periodic[:self.cvs], dtype=bool)

        self.cv_per = cv_per
        
        t = 0
        self.hills = np.loadtxt(name, dtype=np.double)
        self.cv = self.hills[:, 1:1+self.cvs]
        self.cv_min = np.min(self.cv, axis=0) - 1e-8
        self.cv_max = np.max(self.cv, axis=0) + 1e-8
        for i in range(self.cvs):
            flag = 0
            if self.periodic[i]:
                if self.cv_per == None:
                    raise ValueError(
                        "cv_per has to be provided for each periodic CV"
                    )
                try:
                    if self.cv_per[flag][0] <= self.cv_per[flag][1]:
                        self.cv_min[i] = self.cv_per[flag][0]
                        self.cv_max[i] = self.cv_per[flag][1]
                        flag += 1
                except IndexError:
                    raise ValueError(
                        "cv_per has to be provided for each periodic CV"
                    )

        self.sigma = self.hills[:, 1+self.cvs:-2]
        self.heights = self.hills[:, -2]
        self.biasf = self.hills[:, -1]
        if ignore_time:
            if time_step == None:
                time_step = self.hills[0][0]
            self.hills[:, 0] = np.arange(
                time_step, time_step*(len(self.hills)+1), time_step
            )
