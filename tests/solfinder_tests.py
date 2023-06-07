from unittest import TestCase

import numpy as np

import solfinder.MCDM as MCDM


class TestTarget(TestCase):
    """
    Test of Target module
    """

    def setUp(self):
        self.target = MCDM.Target()

    def test_rel_change_zero(self):
        v_test = np.ones(10)
        self.assertAlmostEqual(self.target.rel_change(v_test).all(), np.zeros(10).all())

    def test_solution_found_with_target_zero(self):
        v_test = np.ones(10)
        self.assertEqual(v_test[self.target.solution_found_with_target(0, v_test)], 1.)

    def test_solution_found_with_target_data_exapmle(self):
        with open(r'./Data_example/POBJ_20180101000005_0_51_28_41.dat', 'r') as f:
            data = np.loadtxt(f, unpack=True)

        index_target_05 = self.target.solution_found_with_target(0.5, data[0])
        self.assertTrue(isinstance(index_target_05, (int, np.integer)))


class TestGRA(TestCase):
    """
    Test of GRA module
    """

    def setUp(self):
        self.gra = MCDM.GRA()

    def test_solution_found_with_GRA_data_exapmle(self):
        with open(r'./Data_example/POBJ_20180101000005_0_51_28_41.dat', 'r') as f:
            data = np.loadtxt(f, unpack=True)

        index_gra = self.gra.solution_found_by_gra(data)
        self.assertTrue(isinstance(index_gra, (int, np.integer)))


class TestTOPSIS(TestCase):
    """
    Test of TOPSIS module
    """

    def setUp(self):
        self.topsis = MCDM.TOPSIS()

    def test_solution_found_with_TOPSIS_data_exapmle(self):
        with open(r'./Data_example/POBJ_20180101000005_0_51_28_41.dat', 'r') as f:
            data = np.loadtxt(f, unpack=True)

        index_topsis = self.topsis.solution_found_by_topsis(data, [0.5, 0.5])
        self.assertTrue(isinstance(index_topsis, (int, np.integer)))


class TestVIKOR(TestCase):
    """
    Test of VIKOR module
    """

    def setUp(self):
        self.vikor = MCDM.VIKOR()

    def test_solution_found_with_VIKOR_data_exapmle(self):
        with open(r'./Data_example/POBJ_20180101000005_0_51_28_41.dat', 'r') as f:
            data = np.loadtxt(f, unpack=True)

        set_indices_vikor, index_vikor = self.vikor.solution_found_by_vikor(data, 0.5, [0.5, 0.5])
        self.assertTrue(isinstance(index_vikor, (int, np.integer)))


class TestVikorTarget(TestCase):
    """
    Test of VikorTarget module
    """

    def setUp(self):
        self.vikor_target = MCDM.VikorTarget()

    def test_solution_found_with_VikorTarget_data_exapmle(self):
        with open(r'./Data_example/POBJ_20180101000005_0_51_28_41.dat', 'r') as f:
            data = np.loadtxt(f, unpack=True)

        index_vikor_target = self.vikor_target.solution_found_by_vikor_target(data, 0.5, [0.5, 0.5], 0, 0.5)
        self.assertTrue(isinstance(index_vikor_target, (int, np.integer)))
