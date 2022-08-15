#!/usr/bin/env python
"""
Read config files
"""
from pathlib import Path
import yamale


def read_config_yaml(yamlfile,schematype):
    """Parse config files.

    The function parses the given file and returns a dictionary with the values.

    Note
    ----
    Sections should be named: siminfo and channels

    Parameters
    ----------
    yamlfile : str
        The name of the file to be read including path.

    Returns
    -------
    objs : dictionay
        Dictionary with name given by [Section] each of which contains.
    """

    # dirname = Path(__file__).expanduser().parent
    # schemafile = dirname / "configschema.yaml"
    schemafile = getschemafile(schematype)
    schema = yamale.make_schema(schemafile)
    data = yamale.make_data(yamlfile)
    d1 = yamale.validate(schema, data)

    return data[0][0]

def getschemafile(schematype):
    schema_dict = {'mapping':'mapschema.yaml'}
    dirname = Path(__file__).expanduser().parent
    schemadir = dirname.joinpath('schema')
    return str(schemadir.joinpath(schema_dict[schematype]))
