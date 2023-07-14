import unittest

# Import the module(s) you want to test
from my_project.utils import my_function

# Define a test class that inherits from unittest.TestCase
class MyFunctionTestCase(unittest.TestCase):

    def setUp(self):
        # Initialize any objects or variables needed for the tests
        pass
    #is there a need for a tearDown function? to be decieded

    def test_my_function(self):
        result = my_function(2, 3)
        self.assertEqual(result, 5)
    
    def test_my_function_with_negative_numbers(self):
        result = my_function(-2, -3)
        self.assertEqual(result, -5)
    
# Run the tests
if __name__ == '__main__':
    unittest.main()
