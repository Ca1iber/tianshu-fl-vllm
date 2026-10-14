# SPDX-License-Identifier: Apache-2.0
"""Read-only reflection of one existing configuration table."""
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import inspect


def reflect_one(name, engine):
    # Preserve tables-only discovery and avoid reflecting an empty selection,
    # for which SQLAlchemy can still inspect the entire database in batches.
    if name not in inspect(engine).get_table_names():
        return None
    base = automap_base()
    base.prepare(autoload_with=engine,
                 reflection_options={'only': [name]})
    return base.classes.get(name)
