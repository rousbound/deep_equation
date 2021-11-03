import unittest
from PIL import Image
from deep_equation import predictor


class TestRandomModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_imgs_a = [Image.open("5_1.png"),Image.open("7_1.png"),Image.open("7_2.png"),Image.open("8_2.png"),Image.open("9_1.png"),Image.open("9_2.png")]
        self.input_imgs_b = [Image.open("5_2.png"),Image.open("5_2.png"),Image.open("5_2.png"),Image.open("8_1.png"),Image.open("7_2.png"),Image.open("7_2.png")]
        self.operators = ["+","-","*","/","+","-","*","/"]

    def test_random_predictor(self):
        """
        Test random prediction outputs. 
        """
        basenet = predictor.RandomModel()

        output = basenet.predict(
            self.input_imgs_a, 
            self.input_imgs_b, 
            operators=self.operators, 
            device='cpu',
        )

        self.validate_output(output)
    
    def test_student_predictor(self):
        """
        Test student prediction outputs. 
        """

        basenet = predictor.StudentModel()

        output = basenet.predict(
            self.input_imgs_a, 
            self.input_imgs_b, 
            operators=self.operators, 
            device='cpu',
        )

        # self.validate_output(output)
        print(output)

    def validate_output(self, output):
        """
        Validate output format.
        """

        # Make sure we got one prediction per input_sample
        self.assertEqual(len(output), len(self.input_imgs_a))
        self.assertEqual(len(self.input_imgs_b), len(self.input_imgs_a))
        self.assertEqual(type(output), list)

        # Make sure that that predictions are floats and not other things
        self.assertEqual(type(float(output[0])), float)
        
        # Ensure that the output range is approximately correct
        for out in output:
            self.assertGreaterEqual(out, -10)
            self.assertLessEqual(out, 100)
