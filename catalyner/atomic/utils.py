import numpy as np

def load_reader(path, format="lammps-dump"):
    if format == "lammps-dump":
        return LAMMPSDumpReader(path, format=format)
    elif format == "netcdf":
        return NetCDFReader(path, format=format)
    else:
        raise NotImplementedError(
            "Format {} not implemented".format(format))


class AtomicPropertyReader:
    """
    A class for reading atomic properties from a file.

    Args:
        path (str): The path to the file to be read.
        format (str, optional): The format of the file. Defaults to "lammps-dump".

    Raises:
        NotImplementedError: If the specified format is not implemented.

    Attributes:
        path (str): The path to the file to be read.
        format (str): The format of the file.
    """

    def __init__(self, path, format="lammps-dump"):
        self.path = path
        self.format = format

    def read_atomic_property(self):
        pass


class LAMMPSDumpReader(AtomicPropertyReader):
    """
    A class for reading atomic properties from LAMMPS dump files.

    Attributes:
        path (str): The path to the LAMMPS dump file.

    Methods:
        read_atomic_property(): Reads all frames from the LAMMPS dump file and returns the atomic properties.

    Raises:
        ValueError: If results keys do not match atomic properties in file.
    """
    def _read_single_frame(self, f, results=None):
        """
        Reads a single frame from the LAMMPS dump file.

        Args:
            f (file): The file object to read from.
            results (dict, optional): The dictionary to store the atomic properties. Defaults to None.

        Returns:
            bool: True if a frame is successfully read, False otherwise.
        """
        if not f.readline():  # ITEM TIMESTEP
            return False
        f.readline()
        f.readline()  # ITEM NUMBER OF ATOMS
        n_atoms = int(f.readline())

        # triclinic = len(f.readline().split()) == 9  # ITEM BOX BOUNDS
        for _ in range(4):
            f.readline()

        indices = np.zeros(n_atoms, dtype=int)
        atom_line = f.readline()  # ITEM ATOMS etc
        attrs = atom_line.split()[2:]  # attributes on coordinate line
        attr_to_col_ix = {x: i for i, x in enumerate(attrs)}

        _has_atomic_properties = any("c_" in ix for ix in attr_to_col_ix)
        atomic_cols = [(ix, attr_to_col_ix[ix])
                       for ix in attr_to_col_ix if "c_" in ix] if _has_atomic_properties else []
        ids = "id" in attr_to_col_ix

        if results is None:
            results = {dim[0]: [] for dim in atomic_cols}
        elif results == {}:
            for dim in atomic_cols:
                results[dim[0]] = []
        elif results.keys() != set([dim[0] for dim in atomic_cols]):
            raise ValueError(
                "Results keys do not match atomic properties in file")

        data = {dim[0]: np.zeros(n_atoms) for dim in atomic_cols}

        for i in range(n_atoms):
            fields = f.readline().split()
            if ids:
                indices[i] = fields[attr_to_col_ix["id"]]
            if _has_atomic_properties:
                for dim in atomic_cols:
                    data[dim[0]][i] = fields[dim[1]]
        order = np.argsort(indices)
        if _has_atomic_properties:
            for dim in atomic_cols:
                data[dim[0]] = data[dim[0]][order]
                results[dim[0]].append(data[dim[0]])
        return True

    def read_atomic_property(self):
        """
        Reads all frames from the LAMMPS dump file and returns the atomic properties.

        Returns:
            dict: A dictionary containing the atomic properties.
        """
        with open(self.path) as f:
            # Initialize results with empty lists for each atomic property
            results = {}

            # Read the rest of the frames
            while self._read_single_frame(f, results):
                continue

        # Convert lists to ndarrays
        for key in results:
            results[key] = np.array(results[key])
        return results


class NetCDFReader(AtomicPropertyReader):
    """A class for reading atomic properties from netCDF files."""

    def read_atomic_property(self):
        try:
            from netCDF4 import Dataset  # type: ignore
        except ImportError:
            raise ImportError("netCDF4 is not installed")

        with Dataset(self.path, 'r') as data:
            return {k: np.asarray(v) for k, v in data.variables.items() if "c_" in k}
