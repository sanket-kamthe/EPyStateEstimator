import sqlite3
import numpy as np
import io
import os
from StateModel import Gaussian


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(blob):
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("NParray", convert_array)


# def adapt_gaussian()


def open_database(db=None):
    if db is None:
        path = os.getcwd()
        db = os.path.join(path, 'temp.db')

    conn = sqlite3.connect(db, detect_types=sqlite3.PARSE_DECLTYPES)
    return conn


def create_dynamics_table(db, name='UNGM'):
    schema = """ CREATE TABLE IF NOT EXISTS {:s}  
                (
                Seed INT,
                t REAL,
                X_true NParray, 
                X_noisy NParray,
                Y_true NParray,
                Y_noisy NParray, 
                UNIQUE (Seed, t)
                )""".format(name)
    db.execute(schema)


dynamic_data_string = "INSERT OR IGNORE INTO {}" \
                      "(Seed, t, X_true, X_noisy, Y_true, Y_noisy)" \
                      " VALUES (?, ?, ?, ?, ?, ?)"


def insert_dynamics_data(db, table_name, data, seed):
    query = dynamic_data_string.format(table_name)
    t = 0
    for datum in data:
        x_true, x_noisy, y_true, y_noisy = datum
        values = (seed, t, x_true, x_noisy, y_true, y_noisy)
        t += 1

        db.execute(query, values)

    # db.commit()
# def create_experiment_table(db, name=)