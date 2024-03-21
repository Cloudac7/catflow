class UnitConversion:
    conversion_dict = {}
    
    def convert_value(self, value, input_unit, input_type, output_unit, convert_factor=None):
        conversion_dict = self.conversion_dict.get(input_type, {})
        if input_unit in conversion_dict and output_unit in conversion_dict:
            input_factor = conversion_dict[input_unit]
            output_factor = conversion_dict[output_unit]
            converted_value = value * (input_factor / output_factor)
            return converted_value, output_unit
        else:
            if convert_factor is not None:
                converted_value = value * convert_factor
                return converted_value, output_unit
            else:
                raise ValueError(f"Conversion factor for {input_unit} to {output_unit} is not available")


class PlumedUnits(UnitConversion):
    conversion_dict = {
        "energy": {
            "kJ/mol": 1.0,
            "kcal/mol": 4.184,
            "J/mol": 0.001,
            "eV": 96.48530749925792,
            "Hartree": 2625.499638,
        },
        "length": {
            "nm": 1.0,
            "A": 0.1,
            "pm": 1e-3,
            "fm": 1e-6,
            "um": 1e3,
            "cm": 1e7,
            "m": 1e9,
            "bohr": 0.052917721067,
        },
        "time": {
            "ps": 1.0,
            "fs": 1e-3,
            "ns": 1e3,
            "atomic": 2.418884326509e-5,
        },
    }


class CP2KUnits:
    conversion_dict = {
        "energy": {
            "Hartree": 1.0,
            "J": 1.0 / 4.35974393937059e-18,
            "eV": 1.0 / 27.2113838565563,
            "kJ/mol": 1.0 / 2625.49961709828,
            "kcal/mol": 1.0 / 627.509468713739,
        },
        "length": {
            "bohr": 1.0,
            "A": 1822.88848426455,
        },
        "time": {
            "atomic": 1.0,
            "fs": 1.0 / 2.41888432650478e-2,
            "ps": 1.0 / 41341.3745757512,
        },
    }
