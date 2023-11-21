# -*- coding: utf-8 -*-
###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/aiidateam/aiida-cp2k   #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence, MutableSequence
from dataclasses import dataclass
from warnings import warn


class Cp2kInput:
    """Transforms dictionary into CP2K input"""

    def __init__(self, params=None):
        """Initializing Cp2kInput object"""
        if not params:
            self._params = {}
        else:
            self._params = deepcopy(params)

    def __getitem__(self, key):
        return self._params[key]

    def add_keyword(self, kwpath, value, override=True, conflicting_keys=None):
        """
        Add a value for the given keyword.
        Args:
            kwpath: Can be a single keyword, a path with `/` as divider for sections & key,
                    or a sequence with sections and key.
            value: the value to set the given key to
            override: whether to override the key if it is already present in the self._params
            conflicting_keys: list of keys that cannot live together with the provided key
            (SOMETHING1/[..]/SOMETHING2/KEY). In case override is True, all conflicting keys will
            be removed, if override is False and conflicting_keys are present the new key won't be
            added.
        """

        if isinstance(kwpath, str):
            kwpath = kwpath.split("/")

        Cp2kInput._add_keyword(kwpath, value, self._params, ovrd=override, cfct=conflicting_keys)

    def render(self):
        output = []
        self._render_section(output, deepcopy(self._params))
        return "\n".join(output)

    def param_iter(self, sections=True):
        """Iterator yielding ((section,section,...,section/keyword), value) tuples"""
        stack = [((k,), v) for k, v in self._params.items()]

        while stack:
            key, value = stack.pop(0)
            if isinstance(value, Mapping):
                if sections:
                    yield (key, value)
                stack += [(key + (k,), v) for k, v in value.items()]
            elif isinstance(value, MutableSequence):  # not just 'Sequence' to avoid matching strings
                for entry in value:
                    stack += [(key, entry)]
            else:
                yield (key, value)

    @staticmethod
    def _add_keyword(kwpath, value, params, ovrd, cfct):
        """Add keyword into the given nested dictionary"""
        conflicting_keys_present = []
        # key/value for the deepest level
        if len(kwpath) == 1:
            if cfct:
                conflicting_keys_present = [key for key in cfct if key in params]
            if ovrd:  # if ovrd is True, set the new element's value and remove the conflicting keys
                params[kwpath[0]] = value
                for key in conflicting_keys_present:
                    params.pop(key)
            # if ovrd is False, I only add the new key if (1) it wasn't present beforeAND (2) it does not conflict
            # with any key that is currently present
            elif not conflicting_keys_present and kwpath[0] not in params:
                params[kwpath[0]] = value

        # the key was not present in the dictionary, and we are not yet at the deepest level,
        # therefore a subdictionary should be added
        elif kwpath[0] not in params:
            params[kwpath[0]] = {}
            Cp2kInput._add_keyword(kwpath[1:], value, params[kwpath[0]], ovrd, cfct)

        # if it is a list, loop over its elements
        elif isinstance(params[kwpath[0]], Sequence) and not isinstance(params[kwpath[0]], str):
            for element in params[kwpath[0]]:
                Cp2kInput._add_keyword(kwpath[1:], value, element, ovrd, cfct)

        # if the key does NOT point to a dictionary and we are not yet at the deepest level,
        # therefore, the element should be replaced with an empty dictionary unless ovrd is False
        elif not isinstance(params[kwpath[0]], Mapping):
            if ovrd:
                params[kwpath[0]] = {}
                Cp2kInput._add_keyword(kwpath[1:], value, params[kwpath[0]], ovrd, cfct)
        # if params[kwpath[0]] points to a sub-dictionary, enter into it
        else:
            Cp2kInput._add_keyword(kwpath[1:], value, params[kwpath[0]], ovrd, cfct)

    @staticmethod
    def _render_section(output, params, indent=0):
        """It takes a dictionary and recurses through.
        For key-value pair it checks whether the value is a dictionary and prepends the key with & (CP2K section).
        It passes the valued to the same function, increasing the indentation. If the value is a list, I assume
        that this is something the user wants to store repetitively
        eg:
        .. highlight::
           dict['KEY'] = ['val1', 'val2']
           ===>
           KEY val1
           KEY val2
           or
           dict['KIND'] = [{'_': 'Ba', 'ELEMENT':'Ba'},
                           {'_': 'Ti', 'ELEMENT':'Ti'},
                           {'_': 'O', 'ELEMENT':'O'}]
           ====>
                 &KIND Ba
                    ELEMENT  Ba
                 &END KIND
                 &KIND Ti
                    ELEMENT  Ti
                 &END KIND
                 &KIND O
                    ELEMENT  O
                 &END KIND
        """

        for key, val in sorted(params.items()):
            # keys are not case-insensitive, ensure that they follow the current scheme
            if key.upper() != key:
                raise ValueError(f"keyword '{key}' not upper case.")

            if isinstance(val, Mapping):
                line = f"{' ' * indent}&{key}"
                if "_" in val:  # if there is a section parameter, add it
                    line += f" {val.pop('_')}"
                output.append(line)
                Cp2kInput._render_section(output, val, indent + 3)
                output.append(f"{' ' * indent}&END {key}")

            elif isinstance(val, Sequence) and not isinstance(val, str):
                for listitem in val:
                    Cp2kInput._render_section(output, {key: listitem}, indent)

            elif isinstance(val, bool):
                val_str = '.TRUE.' if val else '.FALSE.'
                output.append(f"{' ' * indent}{key} {val_str}")

            else:
                output.append(f"{' ' * indent}{key} {val}")

    @staticmethod
    def _convert_to_dict(lines):
        pass


@dataclass(frozen=True)
class InsertValue:
    key: str
    value: str or dict


def flatten_seq(sequence):
    for item in sequence:
        if type(item) is list:
            for subitem in flatten_seq(item):
                yield subitem
        else:
            yield item


class Cp2kInputToDict(object):
    # class from ecint
    def __init__(self, filename):
        self.filename = filename
        self.set_var_val = {}
        self.if_flag, self.end_flag = True, True

    def __repr__(self):
        return ''.join(self.lines)

    @property
    def lines(self):
        with open(self.filename) as f:
            lines = f.readlines()
        return lines

    @property
    def well_defined_lines(self):
        return self._get_well_defined_lines(self.lines)

    def get_config(self):
        tree = self.get_tree()
        force_eval = tree["FORCE_EVAL"]
        try:
            if isinstance(force_eval, dict):
                force_eval.pop("SUBSYS")
            elif isinstance(force_eval, list):
                for one_force_eval in force_eval:
                    one_force_eval.pop("SUBSYS")
        except KeyError:
            pass
        return tree

    def extract_kind_section(self):
        force_eval = self.get_tree()["FORCE_EVAL"]
        if isinstance(force_eval, list):
            force_eval = force_eval[0]
        try:
            kind_section_list = force_eval["SUBSYS"]["KIND"]
            kind_section_dict = {kind_section.pop('_'): kind_section for
                                 kind_section in kind_section_list}
            return kind_section_dict
        except KeyError:
            warn('No &KIND info found, so kind section will not be parsed',
                 Warning)

    def get_tree(self):
        return self.get_tree_from_lines(self.well_defined_lines)

    @classmethod
    def get_tree_from_lines(cls, well_defined_lines):
        tree = {}
        for line in well_defined_lines:
            if line.upper().startswith("&END"):
                break
            elif line.upper().startswith("&"):
                name = line.split(None, 1)[0][1:].upper()
                cls._parse_section_start(line, tree)
                if isinstance(tree[name], dict):
                    tree[name].update(
                        cls.get_tree_from_lines(well_defined_lines))
                elif isinstance(tree[name], list):
                    tree[name][-1].update(
                        cls.get_tree_from_lines(well_defined_lines))
            else:
                cls._parse_keyword(line, tree)
        return tree

    @classmethod
    def _parse_section_start(cls, section_start, tree):
        section_pair = section_start.split(None, 1)
        name = section_pair[0][1:].upper()
        section_init = {'_': section_pair[1]} if len(section_pair) == 2 else {}
        tree = cls._update_tree(tree, InsertValue(name, section_init))
        return tree

    @classmethod
    def _parse_keyword(cls, keyword, tree):
        keyword_pair = keyword.split(None, 1)
        name = keyword_pair[0].upper()
        value = keyword_pair[1] if len(keyword_pair) == 2 else ''
        tree = cls._update_tree(tree, InsertValue(name, value))
        return tree

    @classmethod
    def _update_tree(cls, tree, insertval):
        name = insertval.key
        value = insertval.value
        if tree.get(name) and isinstance(tree[name], type(value)):
            tree[name] = [tree[name], value]
        elif tree.get(name) and isinstance(tree[name], list):
            tree[name].append(value)
        else:
            tree[name] = value
        return tree

    def _get_well_defined_lines(self, lines):
        # parse single line
        lines = self._flatten_lines(lines)
        # clean blank lines
        well_defined_lines = filter(None, lines)
        return well_defined_lines

    def _flatten_lines(self, lines):
        lines = list(flatten_seq(lines))
        lines = list(map(self._parse_line, lines))
        if any(type(line) is list for line in lines):
            return self._flatten_lines(lines)
        return lines

    def _parse_line(self, line):
        line = self._remove_comment(line)
        # convert SET
        if line.upper().startswith("@SET"):
            self._convert_set(line)
            line = ''
        else:
            line = self._convert_var(line)
        # convert IF
        if line.upper().startswith("@IF"):
            if not self.end_flag:
                raise ValueError("Do not use nested @IF")
            self.if_flag, self.end_flag = self._convert_if(line), False
            line = ''
        elif line.upper().startswith("@ENDIF"):
            if self.end_flag:
                raise ValueError("Can not find @IF before @ENDIF")
            self.if_flag, self.end_flag = True, True
            line = ''
        if not self.if_flag:
            line = ''
        # convert INCLUDE
        if line.upper().startswith("@INCLUDE"):
            line = self._convert_include(line)
        return line

    @classmethod
    def _remove_comment(cls, line):
        return line.split('!', 1)[0].split('#', 1)[0].strip()

    def _convert_set(self, line):
        variable, value = line.split(None, 2)[1:]
        self.set_var_val.update({variable: value})

    def _convert_var(self, line):
        user_var = re.search(r'\$(\{)?(?P<name>\w+)(?(1)\}|)', line)
        if user_var and (user_var['name'] not in self.set_var_val):
            raise ValueError(f'Variable {user_var} used before defined')
        for variable, value in self.set_var_val.items():
            line = re.sub(r'\$(\{)?(%s)(?(1)\}|)' % variable, value, line)
        return line

    @classmethod
    def _convert_if(cls, line):
        if len(line.split(None, 1)) == 1:
            if_express = False
        else:
            if_express = False if line.split(None, 1)[1] == '0' else True
        return if_express

    @classmethod
    def _convert_include(cls, line):
        filename = line.split(None, 1)[1]
        try:
            with open(filename, 'r') as f:
                file_lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f'No @INCLUDE File: {filename}')
        return [cls._remove_comment(line) for line in file_lines]

def lagrange_mult_log_parser(filename):
    """Collect the Lagrange multipliers from the log file.

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    from scipy.constants import physical_constants

    eV = physical_constants['atomic unit of electric potential'][0]
    a = physical_constants['atomic unit of length'][0]
    eV_a = eV / (a * 1e10)

    # Read the log file
    forces = []
    with open(filename) as f:
        line = 1
        while line:
            line = f.readline()
            if line:
                forces.append(float(line.split()[-1]) * eV_a)
            # just jump the 'Rattle' line
            f.readline() 
    return forces
