# **Demonstration of SU(2) Spin Basis Transformation:**  *Coupling Two Spin-1/2 Systems from an Uncoupled Tensor Product Basis to a Clebsch-Gordan Coupled Basis*

This demo illustrates how a DMRG algorithm transforms the description of two physical sites (a "Left Block" and a "Site s") from an easily formed but symmetry-unaware basis (the simple Cartesian product of their states) into a symmetry-adapted basis that uses Clebsch-Gordan coefficients to explicitly define total spin multiplets (singlets and triplets).

Transformation matrix U from block L and site s in [./transf_matrix_CPU.cu](./transf_matrix_CPU.cu)