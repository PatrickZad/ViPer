from psd2.evaluation import evaluator
import os

from .. import mappers
from ..catalog import DatasetCatalog, MapperCatalog, MetadataCatalog

from .cuhk_sysu import load_cuhk_sysu
from .cuhk_sysu_tbps import load_cuhk_sysu_tbps
from .prw import load_prw
from .prw_tbps import load_prw_tbps


# TODO change evaluator type to "query"
def register_cuhk_sysu_all(datadir):
    name = "CUHK-SYSU_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    name = "CUHK-SYSU_" + "Gallery"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_" + "GalleryGT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG50GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG100"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG100GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG4000GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")


def register_cuhk_sysu_tbps_all(datadir):
    name = "CUHK-SYSU_TBPS_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    name = "CUHK-SYSU_TBPS_" + "Gallery"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_TBPS_" + "GalleryGT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_TBPS_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_TBPS_" + "TestG50GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_TBPS_" + "TestG100"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_TBPS_" + "TestG100GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_TBPS_" + "TestG500"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG500"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_TBPS_" + "TestG1000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG1000"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_TBPS_" + "TestG2000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG2000"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_TBPS_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_TBPS_" + "TestG4000GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu_tbps(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")



def register_prw_all(datadir):
    name = "PRW_Train"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwMapper)
    name = "PRW_Query"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_QueryGT"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_Gallery"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "PRW_GalleryGT"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")

def register_prw_tbps_all(datadir):
    name = "PRW_TBPS_Train"
    DatasetCatalog.register(name, lambda: load_prw_tbps(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwTBPSMapper)
    name = "PRW_TBPS_TrainXi"
    DatasetCatalog.register(name, lambda: load_prw_tbps(datadir, "TrainXi"))
    MapperCatalog.register(name, mappers.PrwTBPSMapper)
    name = "PRW_TBPS_Query"
    DatasetCatalog.register(name, lambda: load_prw_tbps(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_TBPS_QueryGT"
    DatasetCatalog.register(name, lambda: load_prw_tbps(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_TBPS_Gallery"
    DatasetCatalog.register(name, lambda: load_prw_tbps(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "PRW_TBPS_GalleryGT"
    DatasetCatalog.register(name, lambda: load_prw_tbps(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwTBPSMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


_root = "Data"
register_cuhk_sysu_all(os.path.join(_root, "cuhk_sysu"))
register_prw_all(os.path.join(_root, "PRW"))
register_cuhk_sysu_tbps_all(os.path.join(_root, "cuhk_sysu"))
register_prw_tbps_all(os.path.join(_root, "PRW"))

