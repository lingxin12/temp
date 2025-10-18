# This script is based on an original implementation by True Price.
# Created by liminghao
import sys
import numpy as np
import sqlite3

IS_PYTHON3 = sys.version_info[0] >= 3

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def camTodatabase():
    import os
    import argparse

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="database.db")
    parser.add_argument("--txt_path", type=str, default="colmap/sparse_cameras.txt")
    # breakpoint()
    args = parser.parse_args()
    if os.path.exists(args.database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(args.database_path)

    def get_table_names(db):
        cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return [table[0] for table in tables]
    
    def print_table_schema(db, table_name):
        cursor = db.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print(f"\nSchema of table '{table_name}':")
        for col in columns:
            print(col)

    def print_table_contents(db, table_name):
        cursor = db.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()

        print(f"\nContents of table '{table_name}':")
        for row in rows:
            camera_id, model, width, height, params, prior = row
            print(f"{camera_id} {model} {width} {height} {blob_to_array(params, np.float64)} {prior}")

    # tables = get_table_names(db)
    # print_table_schema(db, "cameras")
    # print_table_contents(db, "cameras")
    # breakpoint()



    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(args.txt_path, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(float(strLists[0]))
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(float(strLists[2]))
                height=int(float(strLists[3]))
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()

    # tables = get_table_names(db)
    # print_table_schema(db, "cameras")
    # print_table_contents(db, "cameras")
    # breakpoint()

    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # print_table_schema(db, "cameras")
    # print_table_contents(db, "cameras")
    # breakpoint()
    # Close database.db.
    db.close()

if __name__ == "__main__":
    import sys,os

    camTodatabase()