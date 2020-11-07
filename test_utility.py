import unittest
from utility import Utility

class TestUtility(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_clean_text(self):
        text = "\nThis is awesome!!! On 21st we have a party \t\t We    should enjoy it. "
        actual = Utility.clean_text(text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual, 'this is awesome on st we have a party we should enjoy it')

    def test_lowercase(self):
        text = "Today we'll have a good weather"
        actual = Utility.lowercase(text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual, "today we'll have a good weather")

    def test_remove_newlines(self):
        text = "\n\nThis is a sample text."
        actual = Utility.remove_newlines(text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual.strip(), "This is a sample text.")

    def test_remove_tabs(self):
        text = "\tThis is a sample text."
        actual = Utility.remove_tabs(text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual.strip(), "This is a sample text.")

    def test_remove_digits(self):
        text = "2021 seems to be a good year for traveling"
        actual = Utility.remove_digits(text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual.strip(), "seems to be a good year for traveling")

    def test_remove_multiple_whitespaces(self):
        text = "Hello    where are you    ?"
        actual = Utility.remove_multiple_whitespaces(text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual.strip(), "Hello where are you ?")

    def test_remove_trailing_whitespaces(self):
        text = " This a sentence with trailing white spaces. "
        actual = Utility.remove_trailing_whitespaces(text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual.strip(), "This a sentence with trailing white spaces.")

    def test_remove_special_chars(self):
        text = "Help me!!!"
        actual = Utility.replace_special_chars(text)
        self.assertTrue(isinstance(actual, str))
        self.assertEqual(actual.strip(), "Help me")