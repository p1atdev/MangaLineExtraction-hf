# copied from ViTImageProcessor (https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/image_processing_vit.py)

"""Image processor class for WD v14 Tagger."""

from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import cv2
from PIL import Image

from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import (
    rescale,
    to_channel_dimension_format,
    _rescale_for_pil_conversion,
    to_pil_image,
)
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


def resize_by_factor(
    image: np.ndarray,
    resize_factor: int,
    resample: PILImageResampling = None,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    return_numpy: bool = True,
):
    """
    Resizes `image` to `(height, width)` specified by `size` using the PIL library.

    Args:
        image (`np.ndarray`):
            The image to resize.
        resize_factor (`int`):
            Value for padding the image to a multiple of the factor.
        resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            The filter to user for resampling.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        return_numpy (`bool`, *optional*, defaults to `True`):
            Whether or not to return the resized image as a numpy array. If False a `PIL.Image.Image` object is
            returned.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `np.ndarray`: The resized image.
    """

    resample = resample if resample is not None else PILImageResampling.BILINEAR

    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    do_rescale = False
    if not isinstance(image, Image.Image):
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(
            image, do_rescale=do_rescale, input_data_format=input_data_format
        )

    assert isinstance(image, Image.Image)

    width, height = (
        int(np.ceil(image.size[0] // resize_factor) * resize_factor),
        int(np.ceil(image.size[1] // resize_factor) * resize_factor),
    )
    # solid image
    new_image = Image.new(image.mode, (width, height), "white")

    # paste original image on top left
    new_image.paste(image)

    if return_numpy:
        new_image = np.array(new_image)
        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        new_image = (
            np.expand_dims(new_image, axis=-1) if new_image.ndim == 2 else new_image
        )
        # The image is always in channels last format after converting from a PIL image
        new_image = to_channel_dimension_format(
            new_image, data_format, input_channel_dim=ChannelDimension.LAST
        )
        # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
        # rescale it back to the original range.
        new_image = rescale(new_image, 1 / 255) if do_rescale else new_image

    return new_image


def greyscale(
    image: np.ndarray,
    data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
    return_numpy: bool = True,
):
    """
    Convert `image` to `greyscale` using the PIL library.

    Args:
        image (`np.ndarray`):
            The image to greyscale.
    Returns:
        `np.ndarray`: The greyscaled image.
    """

    if not isinstance(image, Image.Image):
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(
            image, do_rescale=do_rescale, input_data_format=input_data_format
        )

    assert isinstance(image, Image.Image)

    # do greyscale
    image = image.convert("L")

    if return_numpy:
        image = np.array(image)

        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        image = np.expand_dims(image, axis=-1) if image.ndim == 2 else image

        # The image is always in channels last format after converting from a PIL image
        image = to_channel_dimension_format(
            image, data_format, input_channel_dim=ChannelDimension.LAST
        )
        # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
        # rescale it back to the original range.
        image = rescale(image, 1 / 255) if do_rescale else image

    return image


class MLEImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MLE image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        resize_factor (`int`, *optional*, defaults to `16`):
            Value for padding the image to a multiple of the factor.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `False`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        resize_factor: int = 16,
        do_greyscale: bool = True,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1.0,
        do_normalize: bool = False,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resize_factor = resize_factor
        self.do_greyscale = do_greyscale
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = (
            image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN[0]
        )
        self.image_std = (
            image_std if image_std is not None else IMAGENET_STANDARD_STD[0]
        )

    def resize(
        self,
        image: np.ndarray,
        resize_factor: int,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            resize_factor (`int`):
                Value for padding the image to a multiple of the factor.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """

        return resize_by_factor(
            image,
            resize_factor=resize_factor,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def greyscale(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
        input_data_format: Optional[
            Union[str, ChannelDimension]
        ] = ChannelDimension.FIRST,
        **kwargs,
    ):
        """
        Convert an image to greyscale.

        Args:
            image (`np.ndarray`):
                Image to greyscale

        Returns:
            `np.ndarray`: The greyscaled image.
        """

        return greyscale(
            image,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        resize_factor: Optional[int] = None,
        do_greyscale: Optional[bool] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resize_factor (`int`, *optional*, defaults to `self.resize_factor`):
                Value for padding the image to a multiple of the factor.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_greyscale = do_greyscale if do_greyscale is not None else self.do_greyscale
        resample = resample if resample is not None else self.resample
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        resize_factor = (
            resize_factor if resize_factor is not None else self.resize_factor
        )

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and resize_factor is None:
            raise ValueError("Resize factor must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [
                self.resize(
                    image=image,
                    resize_factor=resize_factor,
                    resample=resample,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        if do_greyscale:
            images = [
                self.greyscale(
                    image=image,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for image in images
            ]
            # the channel would be set to 1, so input data format could't be estimated
            input_data_format = ChannelDimension.FIRST

        if do_rescale:
            images = [
                self.rescale(
                    image=image,
                    scale=rescale_factor,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        images = [
            to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )
            for image in images
        ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
