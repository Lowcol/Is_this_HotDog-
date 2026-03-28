import yaml
import numpy as np

import unittest

from src.pipelines import utils


class TestUtils(unittest.TestCase):
    """Test util functions."""
    @classmethod
    def setUpClass(cls):
        with open('config/test_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)

        cls.configs = configs

    def test_label_img_cat(self):
        res = utils.label_img('hotdog')
        self.assertEqual(res, 1)

    def test_label_img_not_cat(self):
        res = utils.label_img('not_hotdog')
        self.assertEqual(res, 0)

    def test_load_data(self):
        res = utils.load_data(train=True, configs=self.configs)
        self.assertGreater(len(res), 0)
        # Each list in the main list has an image, label, and path
        self.assertEqual(len(res[0]), 3)
        # Check that the image looks correct
        self.assertEqual(res[0][0].shape[0], self.configs['image_size'])
        self.assertEqual(res[0][0].shape[1], self.configs['image_size'])
        self.assertEqual(res[0][0].shape[2], 3)
        self.assertIsInstance(res[0][0], np.ndarray)
        # Check that the labels are correct
        self.assertIn(res[0][1], [0, 1])
        self.assertIsInstance(res[0][1], int)
        # Check that the path is correct
        self.assertIsInstance(res[0][2], str)

    def test_format_data_for_model(self):
        dat_list = utils.load_data(
            train=True,
            configs=self.configs
        )
        images, labels, paths = utils.format_data_for_model(
            dat_list=dat_list,
            configs=self.configs
        )

        self.assertGreater(len(images), 0)
        self.assertEqual(images.shape[-1], 3)
        self.assertLessEqual(np.max(images), 1)
        self.assertGreaterEqual(np.min(images), 0)
        self.assertEqual(len(labels), len(images))
        self.assertEqual(len(paths), len(images))
        self.assertTrue(set(np.unique(labels)).issubset({0, 1}))
        self.assertEqual(labels.mean(), 0.5)


if __name__ == '__main__':
    unittest.main()
