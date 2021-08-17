from collections import defaultdict
from typing import List
import re

import albumentations
import cv2
import nltk
import torch

import text_utils

def update_params(self, params, **kwargs):
    if hasattr(self, "interpolation"):
        params["interpolation"] = self.interpolation
    if hasattr(self, "fill_value"):
        params["fill_value"] = self.fill_value
    if hasattr(self, "mask_fill_value"):
        params["mask_fill_value"] = self.mask_fill_value
    #params.update({"cols": kwargs["image"].shape[1], "rows": kwargs["image"].shape[0]})
    return params
albumentations.BasicTransform.update_params = update_params

from albumentations.core.serialization import SERIALIZABLE_REGISTRY, instantiate_lambda
class SetCompose(albumentations.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, 
        additional_targets=None, p=1.0, save_key="replay"
    ):
        super().__init__(transforms, bbox_params, keypoint_params, additional_targets, p)

        self.save_key = save_key
        self.set_deterministic(True, save_key=self.save_key)
        self._target_sets = {}

    def set_deterministic(self, flag, save_key="replay"):
        for t in self.transforms:
            if not t.targets_as_params:
                t.set_deterministic(flag, save_key)

    @property
    def target_sets(self):
        return self._target_sets
    @target_sets.setter
    def target_sets(self, target_sets):
        self._target_sets = target_sets
        self._inv_targets = {v:k for k,v in target_sets.items()}

    def __call__(self, *args, force_apply=False, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")

        transforms = self.transforms
        kwargs_targets = set([self.target_sets[k] for k in kwargs])

        for idx, t in enumerate(transforms):
            result = defaultdict(list)

            t_targets = t.targets
            common_targets = set(t_targets) & kwargs_targets

            kwargs[self.save_key] = defaultdict(dict)


            if len(common_targets) > 1:
                # Maybe TODO a check about t.targets_as_params ?
                target_set_names = {
                    self._inv_targets[target_set_name] 
                    for target_set_name in common_targets
                }

                replayable_t = None

                for target_name in common_targets:
                    target_set_name = self._inv_targets[target_name]
                    target_set = kwargs[target_set_name]

                    for element in target_set:
                        if replayable_t is None:
                            t_result = t(**{
                                target_name: element, 
                                'replay': defaultdict(dict)
                            })
                            
                            serialized = t.get_dict_with_id()
                            self.fill_with_params(serialized, t_result[self.save_key])
                            self.fill_applied(serialized)
                            t_result[self.save_key] = serialized

                            replayable_t = SetCompose._restore_for_replay(t_result[self.save_key])
                        else:
                            t_result = replayable_t(force_apply=True, **{target_name: element})

                        result[target_set_name].append(t_result[target_name])

            elif len(common_targets) == 1:
                target_name = common_targets.pop()

                target_set_name = self._inv_targets[target_name]
                target_set = kwargs[target_set_name]

                for element in target_set:
                    t_result = t(**{target_name: element, 'replay': defaultdict(dict)})
                    result[target_set_name].append(t_result[target_name])

            #for target in common_targets:
            #    kwargs.update(target, result[target])
            kwargs.update(result)

        return kwargs


    @staticmethod
    def _restore_for_replay(transform_dict, lambda_transforms=None):
        """
        Args:
            transform (dict): A dictionary with serialized transform pipeline.
            lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
                This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
                in that dictionary should be named same as `name` arguments in respective lambda transforms from
                a serialized pipeline.
        """
        transform = transform_dict
        applied = transform["applied"]
        params = transform["params"]
        lmbd = instantiate_lambda(transform, lambda_transforms)
        if lmbd:
            transform = lmbd
        else:
            name = transform["__class_fullname__"]
            args = {k: v for k, v in transform.items() if k not in ["__class_fullname__", "applied", "params"]}
            cls = SERIALIZABLE_REGISTRY[name]
            if "transforms" in args:
                args["transforms"] = [
                    SetCompose._restore_for_replay(t, lambda_transforms=lambda_transforms)
                    for t in args["transforms"]
                ]
            transform = cls(**args)

        transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized, all_params):
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized):
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    @staticmethod
    def _restore_for_replay(transform_dict, lambda_transforms=None):
        """
        Args:
            transform (dict): A dictionary with serialized transform pipeline.
            lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
                This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
                in that dictionary should be named same as `name` arguments in respective lambda transforms from
                a serialized pipeline.
        """
        transform = transform_dict
        applied = transform["applied"]
        params = transform["params"]
        lmbd = instantiate_lambda(transform, lambda_transforms)
        if lmbd:
            transform = lmbd
        else:
            name = transform["__class_fullname__"]
            args = {k: v for k, v in transform.items() if k not in ["__class_fullname__", "applied", "params"]}
            cls = SERIALIZABLE_REGISTRY[name]
            if "transforms" in args:
                args["transforms"] = [
                    SetCompose._restore_for_replay(t, lambda_transforms=lambda_transforms)
                    for t in args["transforms"]
                ]
            transform = cls(**args)

        transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

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
    def get_transform_init_args_names(self):
        return ()


class TextNormalization(TextOnlyTransform):
    def apply_to_text(self, text: str, **params) -> str:
        return text.strip().lower()
    def get_transform_init_args_names(self):
        return ()

class NltkTokenization(TextOnlyTransform):
    def apply_to_text(self, text: str, **params) -> List[str]:
        return nltk.word_tokenize(text)
    def get_transform_init_args_names(self):
        return ()

class StartEndPadding(TextOnlyTransform):
    def apply_to_text(self, text: List[str], **params) -> List[str]:
        return [text_utils.start_token] + text + [text_utils.end_token]
    def get_transform_init_args_names(self):
        return ()

class VocabularyEncoding(TextOnlyTransform):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def apply_to_text(self, text: List[str], **params) -> List[int]:
        return self.encoder(text)
    def get_transform_init_args_names(self):
        return ()

class IndicesToTensor(TextOnlyTransform):
    def apply_to_text(self, text: List[int], **params) -> torch.LongTensor:
        return torch.LongTensor(text)
    def get_transform_init_args_names(self):
        return ()
