import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from illeism_analysis import *


class TestAnalyzeModules(unittest.TestCase):

    def test_count_illeism(self):
        text: str = "ひよたんはひよりさんが好きだと言った。"
        self.assertEqual(count_illeism(text), 2)

        text: str = "ひよりっちがひよりに会いに行った。"
        self.assertEqual(count_illeism(text), 2)

        text: str = "ひよりさんはひよりっちが好きだと言った。"
        self.assertEqual(count_illeism(text), 2)

        text: str = "ひよりさんはひよりに会いに行った。"
        self.assertEqual(count_illeism(text), 2)

        text: str = "ひよりさんはひよりに会いに行った。ひよりさんはひよりに会いに行った。"
        self.assertEqual(count_illeism(text), 4)

        text: str = "ひよりさんはひよりに会いに行った。ひよりさんはひよりに会いに行った。ひよりさんはひよりに会いに行った。"
        self.assertEqual(count_illeism(text), 6)

    def test_firstperson_count(self):
        text: str = "私はひよりさんが好きだと言った。"
        self.assertEqual(count_firstperson(text), 1)

        text: str = "私はひよりに会いに行った。"
        self.assertEqual(count_firstperson(text), 1)

        text: str = "わたしはひよりさんが好きだと言った。"
        self.assertEqual(count_firstperson(text), 1)

        text: str = "わたしはひよりに本を渡しにいった。"
        self.assertEqual(count_firstperson(text), 1)

        text: str = "あたしはひよりさんが好きだと言った。"
        self.assertEqual(count_firstperson(text), 1)


if __name__=="__init__":
    unittest.main()