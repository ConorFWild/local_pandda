import os
from pathlib import Path

from sqlalchemy import (Boolean, Column, Float, ForeignKey, Integer, String,
                        create_engine, func)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker


class DatabaseConstants:
    COMPOUND_TABLE = "compound"
    SMILES_TABLE = "smiles"
    REFLECTIONS_TABLE = "reflections"
    RESOLUTION_TABLE = "resolution"
    SPACEGROUP_TABLE = "spacegroup"
    UNIT_CELL_TABLE = "unit_cell"
    MODEL_TABLE = "model"
    DATASET_TABLE = "dataset"
    SYSTEM_TABLE = "system"
    SEQUENCE_IDENTITY_TABLE = "system_identity"
    PANDDA_TABLE = "pandda"
    PANDDA_ERROR_TABLE = "pandda_error"
    EVENT_TABLE = "event"
    REFERENCE_TABLE = "reference"
    AUTOBUILD_TABLE = "autobuild"
    AUTOBUILD_BEST_TABLE = "autobuild_best"
    AUTOBUILD_SKELETON_SCORE_TABLE = "autobuild_skeleton_score"
    AUTOBUILD_RSCC_TABLE = "autobuild_rscc"
    AUTOBUILD_RMSD_TABLE = "autobuild_rmsd"
    AUTOBUILD_HUMAN_TABLE = "autobuild_human"
    REAL_SPACE_CLUSTERING_TABLE = "real_space_clustering"
    EVENT_SCORE_TABLE = "event_score"
    TEST_TABLE = "test"
    TRAIN_TABLE = "train"
    MARKER = "marker"
    MAXIMA = "maxima"


base = declarative_base()


class SmilesRecord(base):
    __tablename__ = DatabaseConstants.SMILES_TABLE
    id = Column(Integer, primary_key=True)
    path = Column(String(255))


class ReflectionsRecord(base):
    __tablename__ = DatabaseConstants.REFLECTIONS_TABLE
    id = Column(Integer, primary_key=True)
    path = Column(String(255))


# class ResolutionRecord(base):
#     __tablename__ = DatabaseConstants.RESOLUTION_TABLE
#     id = Column(Integer, primary_key=True)
#     resolution = Column(Float)
#
#     reflections_id = Column(Integer, ForeignKey(ReflectionsRecord.id))
#     reflections = relationship(ReflectionsRecord)

#
# class SpacegroupRecord(base):
#     __tablename__ = DatabaseConstants.SPACEGROUP_TABLE
#     id = Column(Integer, primary_key=True)
#     spacegroup = Column(Integer)
#
#     reflections_id = Column(Integer, ForeignKey(ReflectionsRecord.id))
#     reflections = relationship(ReflectionsRecord)

#
# class UnitCellRecord(base):
#     __tablename__ = DatabaseConstants.UNIT_CELL_TABLE
#     id = Column(Integer, primary_key=True)
#     a = Column(Float)
#     b = Column(Float)
#     c = Column(Float)
#     alpha = Column(Float)
#     beta = Column(Float)
#     gamma = Column(Float)
#
#     reflections_id = Column(Integer, ForeignKey(ReflectionsRecord.id))
#     reflections = relationship(ReflectionsRecord)
#

class ModelRecord(base):
    __tablename__ = DatabaseConstants.MODEL_TABLE
    id = Column(Integer, primary_key=True)
    path = Column(String(255))


class DatasetRecord(base):
    __tablename__ = DatabaseConstants.DATASET_TABLE
    id = Column(Integer, primary_key=True)
    dtag = Column(String(255))

    # Foriegn keys
    reflections_id = Column(Integer, ForeignKey(ReflectionsRecord.id))
    model_id = Column(Integer, ForeignKey(ModelRecord.id))
    smiles_id = Column(Integer, ForeignKey(SmilesRecord.id))

    # Relationships
    reflections = relationship(ReflectionsRecord)
    model = relationship(ModelRecord)
    smiles = relationship(SmilesRecord)


class MarkerRecord(base):
    __tablename__ = DatabaseConstants.MARKER
    id = Column(Integer, primary_key=True)

    # Data
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)


class MaximaRecord(base):
    __tablename__ = DatabaseConstants.MAXIMA
    id = Column(Integer, primary_key=True)

    # Data
    bdc = Column(Float)
    correlation = Column(Float)
    rotation_x = Column(Float)
    rotation_y = Column(Float)
    rotation_z = Column(Float)
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    mean_map_correlation = Column(Float)
    mean_map_max_correlation = Column(Float)

    # Foriegn keys
    dataset_id = Column(Integer, ForeignKey(DatasetRecord.id))
    marker_id = Column(Integer, ForeignKey(MarkerRecord.id))

    # Relationships
    dataset = relationship(DatasetRecord)
    marker = relationship(MarkerRecord)


class EventRecord(base):
    __tablename__ = DatabaseConstants.EVENT_TABLE
    id = Column(Integer, primary_key=True)

    # Data
    bdc = Column(Float)
    score = Column(Float)
    fragment_size = Column(Float)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)

    # Foriegn keys
    dataset_id = Column(Integer, ForeignKey(DatasetRecord.id))
    marker_id = Column(Integer, ForeignKey(MarkerRecord.id))

    # Relationships
    dataset = relationship(DatasetRecord)
    marker = relationship(MarkerRecord)




class Database:

    def __init__(self, database_path: Path, overwrite: bool = False) -> None:
        super().__init__()
        # conn = sqlite3.connect('example.db')
        if overwrite:
            if database_path.exists():
                os.remove(str(database_path))

        engine = create_engine(f'sqlite:///{str(database_path)}')

        base.metadata.bind = engine
        base.metadata.create_all(engine)

        DBSession = sessionmaker()
        DBSession.bind = engine
        session = DBSession()

        self.session = session
