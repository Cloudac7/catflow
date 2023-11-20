def lammps_variable_parser(input_file):
    from pymatgen.io.lammps.inputs import LammpsInputFile

    lammps_input = LammpsInputFile.from_file(input_file)
    keys, values = [], []
    for k, v in lammps_input.stages[0]['commands']:
        if k == "variable":
            keys.append(v.split()[0])
            try:
                values.append(float(v.split()[2]))
            except ValueError:
                values.append(v.split()[2])
    variable_dict = {k: v for k, v in zip(keys, values)}
    return variable_dict