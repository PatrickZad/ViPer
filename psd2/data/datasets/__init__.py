# Copyright (c) Facebook, Inc. and its affiliates.
from . import builtin as _builtin  # ensure the builtin datasets are registered
from .cuhk_sysu import load_cuhk_sysu
from .cuhk_sysu import subset_names as cuhk_sysu_subsets
from .prw import load_prw
from .prw import subsets as prw_subsets


__all__ = [k for k in globals().keys() if not k.startswith("_")]
