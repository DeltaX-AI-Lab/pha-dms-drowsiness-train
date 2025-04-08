import os
import torch
import numpy as np

from PIL import Image
from icecream import ic
from torch.utils import data
from torchvision import transforms
from typing import List, Tuple, Dict, Optional

from utils import util


class Dataset(data.Dataset):
    def __init__(self, config: Dict, evaluate: bool = False):
        # Store basic configurations and mode
        self.config = config
        self.evaluate = evaluate
        
        # Load core configurations with default values
        self.input_size = config.get('input_size', 256)

        # Dataset root, train-test, mean-landmarks paths
        dataset_path = config.get("dataset_path")
        txt_file_path = config.get("test_txt_path") if evaluate else config.get("train_txt_path")
        
        # Initialize sample data and mean landmarks
        self.samples = self.load_samples(dataset_path, txt_file_path)
        
        # Initialize image transformations
        self.resize = transforms.Resize((self.input_size, self.input_size))
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
        self.transforms = (util.RandomSandEffect(density=0.005, intensity=125, grain_size=(0, 0), p=0.5),
                           util.RandomHSV(p=0.5),
                           util.RandomRotate(angle=45, p=0.5),
                           util.RandomCutOut(p=0.5),
                           util.RandomRGB2IR(p=0.5),
                           util.RandomTranslate(p=0.5),
                           util.RandomFlip(config.get("selected_indices"), p=0.5),
                           util.RandomGaussianBlur(p=0.75))

        # self.show_samples(self.samples)
        # exit()
        
    def __len__(self) -> int:
        """ Returns the number of samples in the dataset. """
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Retrieves the processed image and corresponding heatmaps and offsets for training or testing.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            The image tensor and a tuple of score, offset_x, offset_y, neighbor_x, and neighbor_y.
        """
        image_path, label = self.samples[index]
        image = self.load_image(image_path)
        
        # Apply evaluation transforms
        if self.evaluate:
            image = self.resize(image)
            image = self.normalize(image)
            return image, label

        # Apply training augmentations
        for transform in self.transforms:
            image, label = transform(image, label)
        
        image = self.resize(image)
        image = self.normalize(image)
        return image, label

    @staticmethod
    def load_image(image_path: str) -> Optional[Image.Image]:
        """
        Loads an image from the specified path and converts it to RGB format.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Optional[Image.Image]: Loaded image in RGB format, or None if loading failed.
        """
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    @staticmethod
    def load_samples(dataset_path: str, txt_file_path: str) -> List[Tuple[str, np.ndarray]]:
        """
        Loads image paths and targets from a txt file.

        Args:
            dataset_path (str): The root path to the dataset directory.
            txt_file_path (str): The path to the txt file containing image paths and landmarks.

        Returns:
            List[Tuple[str, np.ndarray]]: A list of tuples where each tuple contains the image path
            and a NumPy array of landmarks.
        """
        samples = []
        
        with open(txt_file_path, 'r') as data:
            for line in data:
                values = line.strip().split()
                
                image_path = os.path.join(dataset_path, values[0])
                target = np.array([float(x) for x in values[1:]]).astype(np.float64)
                
                # Subset the target if it contains 136 values (68 landmarks with x and y)
                if target.shape[0] == 136:
                    target = target[36*2:48*2]  # focusing on eye landmarks
                
                samples.append((image_path, target))
        return samples

    def show_samples(self, samples):
        import cv2
        from icecream import ic
        
        for (image_path, target) in samples:
            # ==========================================
            if "42dot" in image_path and "open" in image_path:
                # Load image
                image = self.load_image(image_path)
                
                # Transform image
                for transform in self.transforms:
                    image, target = transform(image, target)
                image = self.resize(image)
                
                # Convert bakc to the nympy array
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Reshape the landmarks
                lmk = target.reshape(-1, 2) * self.input_size
                lmk = np.round(lmk).astype(np.int32)

                # leftEye = lmk[6:6+6]
                # rightEye = lmk[0:6]

                # Form convexhulls for both eyes
                # leftEyeHull = cv2.convexHull(leftEye)
                # rightEyeHull = cv2.convexHull(rightEye)

                # leftEyeHull = np.round(leftEyeHull).astype(np.int32)
                # rightEyeHull = np.round(rightEyeHull).astype(np.int32)

                # cv2.drawContours(image, [leftEyeHull], -1, (0,255,100), 1)
                # cv2.drawContours(image, [rightEyeHull], -1, (255,255,255), 1)

                for i, (x, y) in enumerate(lmk):
                    cv2.circle(image, (x, y), 1, (0, 255, 0), 1)

                # cv2.imwrite("face.jpg", image)
                cv2.imshow("face", image)
                
                # WaitKey
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    exit()
            
                ic(lmk)
                ic(image_path)
            else:
                pass
            # ==========================================
        cv2.destroyAllWindows()
        exit()
        
    