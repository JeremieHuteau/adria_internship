from typing import Any, Callable, Dict, List

class MetaFactory(type):
    def __getitem__(self, key):
        return self.PRODUCTS[key]

class Factory(metaclass=MetaFactory):
    """
        Copied from https://github.com/kdexd/virtex
    """
    r"""
    Base class for all factories. All factories must inherit this base class
    and follow these guidelines for a consistent behavior:
    * Factory objects cannot be instantiated, doing ``factory = SomeFactory()``
      is illegal. Child classes should not implement ``__init__`` methods.
    * All factories must have an attribute named ``PRODUCTS`` of type
      ``Dict[str, Callable]``, which associates each class with a unique string
      name which can be used to create it.
    """
    PRODUCTS: Dict[str, Callable] = {}

    def __init__(self):
        raise ValueError(
                f"""Cannot instantiate {self.__class__.__name__} object, use
                `create` classmethod to create a product from this factory.
                """)

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        r"""Create an object by its name, args and kwargs."""
        if name not in cls.PRODUCTS:
            raise KeyError(f"{cls.__class__.__name__} cannot create {name}.")

        return cls[name](*args, **kwargs)

    #"""
    #* All factories must implement one classmethod, :meth:`from_config` which
    #  contains logic for creating an object directly by taking name and other
    #  arguments directly from :class:`~virtex.config.Config`. They can use
    #  :meth:`create` already implemented in this base class.
    #* :meth:`from_config` should not use too many extra arguments than the
    #  config itself, unless necessary (such as model parameters for optimizer).
    #@classmethod
    #"""
    #def from_config(cls, config: Config) -> Any:
    #    r"""Create an object directly from config."""
    #    raise NotImplementedError

import coco_captions_datamodule
class DataModuleFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            "CocoCaptions": coco_captions_datamodule.CocoCaptionsDataModule,
    }

import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import transforms
class ImageTransformFactory(Factory):
    def _RandomRotation(degrees: int, **kwargs):
        return alb.Rotate(limit=degrees, border_mode=cv2.BORDER_CONSTANT, 
                **kwargs)
    def _GridDropout(**kwargs):
        return alb.GridDropout(random_offset=True,
                **kwargs)

    PRODUCTS: Dict[str, Callable] = {
            'ToTensor': ToTensorV2,

            'Resize': alb.Resize,
            'CenterCrop': alb.CenterCrop,
            'RandomCrop': alb.RandomCrop,
            'RandomResizedCrop': alb.RandomResizedCrop,

            'HorizontalFlip': transforms.HorizontalFlip,
            'RandomRotation': _RandomRotation,
            'ColorJitter': alb.ColorJitter,
            'GridDropout': _GridDropout,

            'Normalization': alb.Normalize,
    }


import pickle
class TextTransformFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            'Normalization': transforms.TextNormalization,
            'NltkTokenization': transforms.NltkTokenization,
            'StartEndPadding': transforms.StartEndPadding,
            'VocabularyEncoding': lambda path: transforms.VocabularyEncoding(
                pickle.load(open(path, 'rb'))),
            'IndicesToTensor': transforms.IndicesToTensor,
    }

class TransformFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            'ImageToTensor': ImageTransformFactory['ToTensor'],
            'Resize': ImageTransformFactory['Resize'],
            'CenterCrop': ImageTransformFactory['CenterCrop'],
            'RandomCrop': ImageTransformFactory['RandomCrop'],
            'RandomResizedCrop': ImageTransformFactory['RandomResizedCrop'],
            'HorizontalFlip': ImageTransformFactory['HorizontalFlip'],
            'RandomRotation': ImageTransformFactory['RandomRotation'],
            'ColorJitter': ImageTransformFactory['ColorJitter'],
            'GridDropout': ImageTransformFactory['GridDropout'],
            'ImageNormalization': ImageTransformFactory['Normalization'],

            'TextNormalization': TextTransformFactory['Normalization'],
            'NltkTokenization': TextTransformFactory['NltkTokenization'],
            'StartEndPadding': TextTransformFactory['StartEndPadding'],
            'VocabularyEncoding': TextTransformFactory['VocabularyEncoding'],
            'IndicesToTensor': TextTransformFactory['IndicesToTensor'],
    }

import vse_models
class VSEModelFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            'VSE': vse_models.VSE,
    }
class VisionModelFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            'BasicCNN': vse_models.BasicCnnEncoder,
            'ResNet': vse_models.ResNetEncoder,
    }
class TextModelFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            'GRU': vse_models.GruEncoder,
    }

import torch.optim
class OptimizerFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
    }
class SchedulerFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            'LinearWarmup': lambda *a, num_warmup_steps: torch.optim.lr_scheduler.LambdaLR(
                *a,
                lambda step: min(step/num_warmup_steps, 1.0)
            ),
            'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

import triplet_margin_loss
import vse_models
class CallbackFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
            'HardestFractionDecay': triplet_margin_loss.HardestFractionDecay,
            'Unfreezing': vse_models.Unfreezing,
    }



