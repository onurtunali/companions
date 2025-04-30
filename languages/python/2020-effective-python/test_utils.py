from unittest import TestCase, main

from utils import to_str


def setUpModule():
    "This is run at the beginning of tests"
    print("== Module setup ==")


def tearDownModule():
    "This is run at the end of tests"
    print("== Module cleanup ==")


class UtilsTestCase(TestCase):
    def setUp(self):
        "For every test case function this function is run first"
        print("\n** Testcase setup **\n")

    def tearDown(self):
        "For every test case function this function is run last"
        print("\n** Testcase cleanup **\n")

    def test_to_str_bytes(self):
        print("Testing bytes")
        self.assertEqual("hello", to_str(b"hello"))

    def test_to_str_str(self):
        print("Testing str")
        self.assertEqual("hello", to_str("hello"))

    def test_to_str_bad(self):
        print("Testing exception")
        with self.assertRaises(TypeError):
            to_str(1)


if __name__ == "__main__":
    main()
