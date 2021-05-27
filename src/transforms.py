from typing import List
import re

import albumentations
import cv2
import nltk
import torch

class TextOnlyTransform(albumentations.BasicTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
    @property
    def targets(self):
        return {'text': self.apply_to_text}
    def apply_to_text(self):
        raise NotImplementedError

class ImageTextTransform(albumentations.BasicTransform):
    @property
    def targets(self):
        return {'image': self.apply, 'text': self.apply_to_text}
    def apply_to_text(self):
        raise NotImplementedError

class HorizontalFlip(ImageTextTransform):
    regex = re.compile(r"\bleft\b|\bright\b")
    substitution = lambda self, m: "right" if m.group() == "left" else "left"

    def apply(self, image, **params):
        return cv2.flip(image, 1)
    def apply_to_text(self, text, **params):
        return self.regex.sub(self.substitution, text)

class TextNormalization(TextOnlyTransform):
    def apply_to_text(self, text: str, **params) -> str:
        return text.strip().lower()

class NltkTokenization(TextOnlyTransform):
    def apply_to_text(self, text: str, **params) -> List[str]:
        return nltk.word_tokenize(text)

class VocabularyEncoding(TextOnlyTransform):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def apply_to_text(self, text: List[str], **params) -> List[int]:
        return self.encoder(text)

class IndicesToTensor(TextOnlyTransform):
    def apply_to_text(self, text: List[int], **params) -> torch.LongTensor:
        return torch.LongTensor(text)
