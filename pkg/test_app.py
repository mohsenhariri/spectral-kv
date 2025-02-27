import unittest


class TestCalc(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Database connection
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        return super().tearDownClass()

    def setUp(self) -> None:
        print("start")
        return super().setUp()

    def tearDown(self) -> None:
        print("finish")
        return super().tearDown()

    def test_app(self):
        pass


if __name__ == "__main__":
    unittest.main()
