# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/quatrex/quatrex/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                        |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| src/qttools/\_\_about\_\_.py                                |        1 |        0 |    100% |           |
| src/qttools/\_\_init\_\_.py                                 |       41 |       17 |     59% |17-26, 42-43, 49-63, 74 |
| src/qttools/boundary\_conditions/\_\_init\_\_.py            |        2 |        0 |    100% |           |
| src/qttools/boundary\_conditions/boundary\_system.py        |       85 |       16 |     81% |102, 104, 107, 112, 114, 116, 121-125, 230, 272-276, 300-303, 360 |
| src/qttools/boundary\_conditions/lyapunov/\_\_init\_\_.py   |        4 |        0 |    100% |           |
| src/qttools/boundary\_conditions/lyapunov/doubling.py       |       25 |       16 |     36% |     59-91 |
| src/qttools/boundary\_conditions/lyapunov/lyapunov.py       |      103 |       27 |     74% |110, 140-158, 209-210, 253, 264, 268, 295, 330-342, 372, 375, 382, 530-531 |
| src/qttools/boundary\_conditions/lyapunov/spectral.py       |       24 |        0 |    100% |           |
| src/qttools/boundary\_conditions/obc/\_\_init\_\_.py        |        4 |        0 |    100% |           |
| src/qttools/boundary\_conditions/obc/obc.py                 |       24 |        3 |     88% |150-151, 194 |
| src/qttools/boundary\_conditions/obc/sancho\_rubio.py       |       28 |       21 |     25% |32-33, 55-88 |
| src/qttools/boundary\_conditions/obc/spectral.py            |      139 |        7 |     95% |95, 445, 448-450, 454, 464 |
| src/qttools/boundary\_conditions/system\_reduction.py       |       16 |        0 |    100% |           |
| src/qttools/comm/\_\_init\_\_.py                            |        3 |        0 |    100% |           |
| src/qttools/comm/comm.py                                    |      325 |      143 |     56% |18, 71-72, 79, 166, 178, 182, 185, 195, 201-240, 245, 266, 272-273, 278, 281, 287, 292-304, 329, 336, 341, 350, 356-368, 399, 405-406, 411, 416, 421, 427-438, 463, 469-470, 474, 477-487, 523, 529, 546, 584, 590-591, 595-600, 609-627, 650-658, 684-706, 729, 736-737, 740, 785, 792-793, 796, 852-873, 910-931, 998, 1006, 1009 |
| src/qttools/datastructures/\_\_init\_\_.py                  |        5 |        0 |    100% |           |
| src/qttools/datastructures/dsdbcoo.py                       |      184 |       12 |     93% |62, 67, 316, 324, 429, 434-439, 442, 445, 549, 623 |
| src/qttools/datastructures/dsdbcsr.py                       |      144 |      123 |     15% |67-93, 97-109, 137-167, 198-213, 231-292, 308-357, 386-394, 416, 467-500, 513-526 |
| src/qttools/datastructures/dsdbsparse.py                    |      322 |       44 |     86% |53, 141, 147, 162, 167, 170, 173, 337-360, 389, 482, 485, 504, 523, 526, 532, 731-735, 741-745, 855, 956, 965, 1090, 1097, 1102, 1128, 1171, 1175, 1179, 1182, 1187 |
| src/qttools/datastructures/routines.py                      |      194 |       19 |     90% |42, 46, 149-150, 166-178, 212, 222, 398, 403, 407, 510 |
| src/qttools/fft/\_\_init\_\_.py                             |        2 |        0 |    100% |           |
| src/qttools/fft/convolve.py                                 |       11 |        9 |     18% |     24-32 |
| src/qttools/fft/ffts.py                                     |       28 |        1 |     96% |         8 |
| src/qttools/greens\_function\_solver/\_\_init\_\_.py        |        5 |        0 |    100% |           |
| src/qttools/greens\_function\_solver/\_serinv.py            |      546 |        8 |     99% |891-899, 997-1004 |
| src/qttools/greens\_function\_solver/inv.py                 |       63 |       53 |     16% |33, 59-83, 129-189 |
| src/qttools/greens\_function\_solver/rgf.py                 |      145 |       31 |     79% |53-109, 161, 172, 176, 181 |
| src/qttools/greens\_function\_solver/rgf\_dist.py           |       79 |       30 |     62% |56-125, 180, 184, 189 |
| src/qttools/greens\_function\_solver/solver.py              |       32 |        0 |    100% |           |
| src/qttools/kernels/\_\_init\_\_.py                         |        5 |        0 |    100% |           |
| src/qttools/kernels/datastructure/\_\_init\_\_.py           |       11 |        5 |     55% |     12-18 |
| src/qttools/kernels/datastructure/cupy/\_\_init\_\_.py      |        2 |        2 |      0% |       5-7 |
| src/qttools/kernels/datastructure/cupy/\_cupy\_jit.py       |       56 |       56 |      0% |     3-238 |
| src/qttools/kernels/datastructure/cupy/\_cupy\_rawkernel.py |       57 |       57 |      0% |     3-160 |
| src/qttools/kernels/datastructure/cupy/dsdbcoo.py           |       57 |       57 |      0% |     5-265 |
| src/qttools/kernels/datastructure/cupy/dsdbcsr.py           |       81 |       81 |      0% |     5-328 |
| src/qttools/kernels/datastructure/cupy/dsdbsparse.py        |       14 |       14 |      0% |      5-52 |
| src/qttools/kernels/datastructure/numba/\_\_init\_\_.py     |        0 |        0 |    100% |           |
| src/qttools/kernels/datastructure/numba/dsdbcoo.py          |       50 |        0 |    100% |           |
| src/qttools/kernels/datastructure/numba/dsdbcsr.py          |       87 |       73 |     16% |31-41, 87-108, 141-167, 194-196, 223-225, 261-312, 338-339 |
| src/qttools/kernels/datastructure/numba/dsdbsparse.py       |       12 |        0 |    100% |           |
| src/qttools/kernels/inplace/\_\_init\_\_.py                 |        7 |        3 |     57% |     12-18 |
| src/qttools/kernels/inplace/cupy/\_\_init\_\_.py            |        2 |        2 |      0% |      5-10 |
| src/qttools/kernels/inplace/cupy/\_cupy\_rawkernel.py       |       29 |       29 |      0% |      3-87 |
| src/qttools/kernels/inplace/cupy/inplace.py                 |       25 |       25 |      0% |     5-107 |
| src/qttools/kernels/inplace/numba/\_\_init\_\_.py           |        2 |        0 |    100% |           |
| src/qttools/kernels/inplace/numba/inplace.py                |       36 |        0 |    100% |           |
| src/qttools/kernels/linalg/\_\_init\_\_.py                  |        7 |        0 |    100% |           |
| src/qttools/kernels/linalg/eig.py                           |      122 |       63 |     48% |11-13, 86, 121-133, 154-181, 192-267, 313, 315, 323, 326, 333, 350, 352, 356 |
| src/qttools/kernels/linalg/eigvalsh.py                      |       32 |        7 |     78% |40, 85, 93, 101-102, 109-110 |
| src/qttools/kernels/linalg/inv.py                           |       14 |        6 |     57% | 13, 18-23 |
| src/qttools/kernels/linalg/kron.py                          |        7 |        0 |    100% |           |
| src/qttools/kernels/linalg/qr.py                            |       36 |       29 |     19% |34-48, 86-113 |
| src/qttools/kernels/linalg/svd.py                           |       45 |        6 |     87% |46-47, 108, 114, 125-126 |
| src/qttools/kernels/operator.py                             |       34 |       26 |     24% |9-59, 151-167 |
| src/qttools/nevp/\_\_init\_\_.py                            |        4 |        0 |    100% |           |
| src/qttools/nevp/beyn.py                                    |      109 |       12 |     89% |169-177, 266, 286-287, 329, 357, 366-372, 380 |
| src/qttools/nevp/full.py                                    |       67 |       34 |     49% |55-73, 101-119, 143, 150-161, 177-179, 188-196 |
| src/qttools/nevp/nevp.py                                    |        5 |        0 |    100% |           |
| src/qttools/profiling/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| src/qttools/profiling/profiler.py                           |      162 |       23 |     86% |27-30, 47-48, 261, 271, 276, 286-288, 381, 384-385, 388, 397-399, 404-406, 410-412 |
| src/qttools/toeplitz/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| src/qttools/toeplitz/circulant.py                           |       45 |        4 |     91% |29, 92, 203, 206 |
| src/qttools/toeplitz/toeplitz.py                            |       88 |        5 |     94% |21, 153, 299, 395, 398 |
| src/qttools/utils/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| src/qttools/utils/gpu\_utils.py                             |       70 |       32 |     54% |10, 56-58, 84-91, 129-134, 144-145, 156, 187-190, 221-224, 261, 299-302, 312, 322, 332-333 |
| src/qttools/utils/hdf5\_utils.py                            |       51 |        7 |     86% |37, 52, 74, 104, 109, 114, 150 |
| src/qttools/utils/inplace\_utils.py                         |       50 |       50 |      0% |     5-149 |
| src/qttools/utils/memory\_utils.py                          |       29 |        5 |     83% | 30, 56-60 |
| src/qttools/utils/mpi\_utils.py                             |       42 |        4 |     90% |67, 94, 110-112 |
| src/qttools/utils/solvers\_utils.py                         |        6 |        0 |    100% |           |
| src/qttools/utils/sparse\_utils.py                          |       57 |        0 |    100% |           |
| src/qttools/utils/stack\_utils.py                           |        7 |        1 |     86% |        27 |
| src/qttools/wave\_function\_solver/\_\_init\_\_.py          |       10 |        0 |    100% |           |
| src/qttools/wave\_function\_solver/auto\_select.py          |       31 |       18 |     42% |42-54, 61-73 |
| src/qttools/wave\_function\_solver/cudss.py                 |       74 |       53 |     28% |7-30, 63-82, 99-129, 148-163, 183-193, 209, 225, 240, 276-307 |
| src/qttools/wave\_function\_solver/mumps.py                 |       32 |       20 |     38% |8, 56-77, 113-126 |
| src/qttools/wave\_function\_solver/pardiso.py               |       33 |        7 |     79% |10-11, 46, 51, 54, 65, 109 |
| src/qttools/wave\_function\_solver/solver.py                |        6 |        0 |    100% |           |
| src/qttools/wave\_function\_solver/superlu.py               |       21 |        2 |     90% |    15, 48 |
| src/qttools/wave\_function\_solver/thomas.py                |       87 |       67 |     23% |13, 18, 47-80, 101-111, 130, 146-174, 190-222, 239-240, 285-302 |
| src/quatrex/\_\_about\_\_.py                                |        1 |        0 |    100% |           |
| src/quatrex/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| src/quatrex/bandstructure/\_\_init\_\_.py                   |        0 |        0 |    100% |           |
| src/quatrex/bandstructure/band\_edges.py                    |      147 |       41 |     72% |20-23, 42-43, 101, 150-151, 250, 253, 300-335, 345, 350, 445-447, 453-455, 552-560 |
| src/quatrex/bandstructure/contact.py                        |       47 |        2 |     96% |  134, 179 |
| src/quatrex/cli/\_\_init\_\_.py                             |        2 |        0 |    100% |           |
| src/quatrex/cli/main.py                                     |      131 |       55 |     58% |44-46, 138-140, 147-149, 170-189, 241, 248-251, 277-303, 333-348, 364, 369 |
| src/quatrex/config/\_\_init\_\_.py                          |        2 |        0 |    100% |           |
| src/quatrex/config/merge.py                                 |       24 |       21 |     12% |     38-83 |
| src/quatrex/core/\_\_init\_\_.py                            |        0 |        0 |    100% |           |
| src/quatrex/core/config.py                                  |      702 |       64 |     91% |1235, 1242, 1260, 1268, 1428, 1525, 1669-1686, 1841, 1857, 1865, 1873, 1881, 1890, 1896, 1902, 1920-1921, 1927, 2093-2097, 2102-2114, 2250-2252, 2377-2383, 2397-2401, 2413-2414, 2432, 2437, 2452, 2457, 2460, 2463, 2477, 2481, 2485, 2502, 2515, 2537 |
| src/quatrex/core/constants.py                               |        9 |        0 |    100% |           |
| src/quatrex/core/observables.py                             |       32 |        1 |     97% |        41 |
| src/quatrex/core/qtbm.py                                    |      447 |       38 |     91% |106, 153-158, 207, 211, 213, 215, 219, 321, 369, 462-464, 500, 739-741, 969-977, 991-999, 1144, 1234 |
| src/quatrex/core/scba.py                                    |      375 |       57 |     85% |68, 78, 151, 177, 180, 308, 351-355, 360-364, 468, 484, 663, 681-715, 726-729, 743-750, 800, 823-825, 831-844 |
| src/quatrex/core/scsp.py                                    |       57 |       12 |     79% |76-81, 104-113, 120, 146-149 |
| src/quatrex/core/sse.py                                     |        4 |        0 |    100% |           |
| src/quatrex/core/statistics.py                              |        6 |        0 |    100% |           |
| src/quatrex/core/subsystem.py                               |       53 |       10 |     81% |80, 99, 120, 139, 182-189, 229-232 |
| src/quatrex/core/transport.py                               |        9 |        0 |    100% |           |
| src/quatrex/core/utils.py                                   |       32 |       12 |     62% |49-51, 62, 117-131 |
| src/quatrex/coulomb\_screening/\_\_init\_\_.py              |        3 |        0 |    100% |           |
| src/quatrex/coulomb\_screening/polarization.py              |       89 |       29 |     67% |20, 44-57, 127, 141-161, 193, 215 |
| src/quatrex/coulomb\_screening/solver.py                    |      241 |        8 |     97% |77, 83, 95, 171, 838-845 |
| src/quatrex/device/\_\_init\_\_.py                          |        3 |        0 |    100% |           |
| src/quatrex/device/contact.py                               |      299 |       17 |     94% |83, 123, 125, 131, 157, 159, 165, 187, 189, 195, 262, 286, 354, 395, 415, 459, 1013 |
| src/quatrex/device/contact\_discovery.py                    |      130 |       12 |     91% |163, 322, 326, 563, 620, 625, 627, 647, 744, 748, 755, 783 |
| src/quatrex/device/device.py                                |      107 |       12 |     89% |147, 183, 239, 247, 254, 265-266, 278, 284, 292, 302-303 |
| src/quatrex/device/inputs.py                                |      202 |       20 |     90% |56, 64, 170, 178, 180, 182, 185, 209, 322, 326, 444, 455, 463, 472, 489, 494, 552-553, 602, 672 |
| src/quatrex/electron/\_\_init\_\_.py                        |        6 |        0 |    100% |           |
| src/quatrex/electron/solver.py                              |      374 |       25 |     93% |263, 349, 353, 360, 367, 374, 378, 382, 385, 390, 557, 606, 625, 642, 649, 788, 792, 829, 878-892, 1169-1174 |
| src/quatrex/electron/sse\_coulomb\_screening.py             |      119 |       18 |     85% |20, 330-351 |
| src/quatrex/electron/sse\_fock.py                           |       26 |        0 |    100% |           |
| src/quatrex/electron/sse\_phonon.py                         |       35 |        3 |     91% |32, 36, 51 |
| src/quatrex/electron/sse\_photon.py                         |        2 |        0 |    100% |           |
| src/quatrex/electrostatics/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| src/quatrex/electrostatics/\_params.py                      |       12 |        0 |    100% |           |
| src/quatrex/electrostatics/assembly.py                      |       73 |       42 |     42% |39, 45-46, 52, 62-65, 149-189, 226-273 |
| src/quatrex/electrostatics/density\_response.py             |       61 |       26 |     57% |35-44, 65-76, 117-126, 148-156, 172-184, 201-214 |
| src/quatrex/electrostatics/electrostatics.py                |      117 |       43 |     63% |72, 110-116, 132-138, 198, 212, 248-282, 303-345 |
| src/quatrex/electrostatics/fermi\_integrals.py              |      117 |       59 |     50% |42, 80, 84, 148-157, 178-220, 255-256, 261-262, 267-268, 273-274, 279-282, 333, 337, 341, 344, 349-356 |
| src/quatrex/electrostatics/geometry\_config.py              |      142 |       21 |     85% |212, 217, 256, 331-338, 374-382, 386, 391, 401, 417, 420, 443 |
| src/quatrex/electrostatics/meshing.py                       |      327 |      239 |     27% |15-16, 54-59, 76-83, 103-111, 132-141, 163-180, 206-213, 236-239, 263-269, 319-345, 370-375, 396-403, 425-430, 471, 487, 502-512, 527-533, 552-562, 580-625, 638-699, 723-724, 743-886, 901-942 |
| src/quatrex/electrostatics/mixer.py                         |       67 |       36 |     46% |44, 63, 142-155, 177-228 |
| src/quatrex/electrostatics/solver.py                        |      106 |       14 |     87% |120-127, 138-149, 184, 190, 243-244, 310-311 |
| src/quatrex/grid/\_\_init\_\_.py                            |        3 |        0 |    100% |           |
| src/quatrex/grid/energies.py                                |       24 |        9 |     62% | 43, 51-69 |
| src/quatrex/grid/kpoints.py                                 |        5 |        0 |    100% |           |
| src/quatrex/phonon/\_\_init\_\_.py                          |        3 |        0 |    100% |           |
| src/quatrex/phonon/polarization.py                          |        2 |        0 |    100% |           |
| src/quatrex/phonon/solver.py                                |        7 |        1 |     86% |        20 |
| src/quatrex/photon/\_\_init\_\_.py                          |        3 |        0 |    100% |           |
| src/quatrex/photon/polarization.py                          |        2 |        0 |    100% |           |
| src/quatrex/photon/solver.py                                |        7 |        1 |     86% |        18 |
| src/quatrex/post\_processing/\_\_init\_\_.py                |        2 |        2 |      0% |       5-7 |
| src/quatrex/post\_processing/plot\_ldos.py                  |       14 |       14 |      0% |      5-35 |
| src/quatrex/pre\_processsing/\_\_init\_\_.py                |        2 |        2 |      0% |       5-7 |
| src/quatrex/pre\_processsing/contact\_bandstructure.py      |      102 |      102 |      0% |     5-292 |
| src/quatrex/pre\_processsing/contact\_fermi\_level.py       |       41 |       41 |      0% |     5-127 |
| src/quatrex/pre\_processsing/pre\_process.py                |       15 |       15 |      0% |      5-39 |
| **TOTAL**                                                   | **9474** | **2589** | **73%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/quatrex/quatrex/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/quatrex/quatrex/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/quatrex/quatrex/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/quatrex/quatrex/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fquatrex%2Fquatrex%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/quatrex/quatrex/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.