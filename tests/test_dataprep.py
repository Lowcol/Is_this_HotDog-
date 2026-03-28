import unittest
from pathlib import Path
from src.dataprep.script import detect_class, unique_destination

class TestDataPrep(unittest.TestCase):
    
    def test_detect_class_exact_match(self):
        """Test if the script correctly extracts the class name from path parts."""
        input_dir = Path("/some/input/dir")
        
        # Test hotdog path
        hotdog_path = input_dir / "hotdog" / "img1.jpg"
        self.assertEqual(detect_class(hotdog_path, input_dir), "hotdog")
        
        # Test pets path
        pets_path = input_dir / "pets" / "img2.jpg"
        self.assertEqual(detect_class(pets_path, input_dir), "pets")

    def test_detect_class_singular_plural_match(self):
        """Test if it maps plural/singular folder variants correctly."""
        input_dir = Path("/data")
        
        person_path = input_dir / "person" / "img.jpg"
        self.assertEqual(detect_class(person_path, input_dir), "people")
        
        pet_path = input_dir / "pet" / "cat.jpg"
        self.assertEqual(detect_class(pet_path, input_dir), "pets")

    def test_detect_class_unknown(self):
        """Test unknown folder structure returns None."""
        input_dir = Path("/data")
        unknown_path = input_dir / "cars" / "honda.jpg"
        self.assertIsNone(detect_class(unknown_path, input_dir))
        
    def test_unique_destination_no_conflict(self):
        """Test that destination generates the exact same path if no conflict."""
        path = Path("/non_existent_folder_abc/test_file.jpg")
        self.assertEqual(unique_destination(path), path)

if __name__ == '__main__':
    unittest.main()
