

from m5.params import *
from m5.SimObject import SimObject


class SystolicMatrixMultiplication(SimObject):
    type = "SystolicMatrixMultiplication"
    cxx_class = "gem5::SystolicMatrixMultiplication"
    cxx_header = "cpu/minor/systolic_m2m.hh"

    kernel_dim = Param.Int(4, "Systolic array dimension")


class SystolicMatrixMultiplication_3D(SimObject):
    type = "SystolicMatrixMultiplication_3D"
    cxx_class = "gem5::SystolicMatrixMultiplication_3D"
    cxx_header = "cpu/minor/systolic_m2m_3D.hh"

    kernel_dim_h = Param.Int(4, "Systolic array height")
    kernel_dim_w = Param.Int(4, "Systolic array width")
    n_3d_layers = Param.Int(2, "Number of 3D layers")