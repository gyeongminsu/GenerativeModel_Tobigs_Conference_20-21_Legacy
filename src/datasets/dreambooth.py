from torch.utils.data import Dataset
import os
import PIL
from PIL import Image
import numpy as np
import random

import torch
from torchvision import transforms

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        data_root,
        tokenizer_1,
        tokenizer_2,
        class_prompt,
        class_data_root=None,
        class_num=None,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.class_data_root = class_data_root
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.class_prompt = class_prompt

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)

        if class_data_root is not None:
            self.class_images_path = [os.path.join(self.class_data_root, file_path) for file_path in os.listdir(self.class_data_root)]
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_images)
        else:
            self.class_data_root = None

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["original_size"] = (image.height, image.width)

        image = image.resize((self.size, self.size), resample=self.interpolation)

        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.crop(image)
        else:
            y1, x1, h, w = self.crop.get_params(image, (self.size, self.size))
            image = transforms.functional.crop(image, y1, x1, h, w)

        example["crop_top_left"] = (y1, x1)

        example["input_ids_1"] = self.tokenizer_1(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_2"] = self.tokenizer_2(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        

        ### for prior preservation
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[i % self.num_class_images])

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_original_size"] = (class_image.height, class_image.width)        
            
            class_image = class_image.resize((self.size, self.size), resample=self.interpolation)
            
            if self.center_crop:
                y1 = max(0, int(round((class_image.height - self.size) / 2.0)))
                x1 = max(0, int(round((class_image.width - self.size) / 2.0)))
                class_image = self.crop(class_image)
            else:
                y1, x1, h, w = self.crop.get_params(class_image, (self.size, self.size))
                class_image = transforms.functional.crop(class_image, y1, x1, h, w)
            example["class_crop_top_left"] = (y1, x1)
            
            example["class_input_ids_1"] = self.tokenizer_1(
                self.class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_1.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

            example["class_input_ids_2"] = self.tokenizer_2(
                self.class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            
            class_image = np.array(class_image).astype(np.uint8)

            class_image = Image.fromarray(class_image)

            class_image = self.flip_transform(class_image)
            class_image = np.array(class_image).astype(np.uint8)
            class_image = (class_image / 127.5 - 1.0).astype(np.float32)

            example["class_pixel_values"] = torch.from_numpy(class_image).permute(2, 0, 1)
        return example
    
    