import os

import PIL.Image
import numpy as np
import torch


class LoadImageInFolder:
    """
        A example node

        Class methods
        -------------
        INPUT_TYPES (dict):
            Tell the main program input parameters of nodes.

        Attributes
        ----------
        RETURN_TYPES (`tuple`):
            The type of each element in the output tulple.
        FUNCTION (`str`):
            The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
        OUTPUT_NODE ([`bool`]):
            If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
            The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
            Assumed to be False if not present.
        CATEGORY (`str`):
            The category the node should appear in the UI.
        execute(s) -> tuple || None:
            The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
            For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
        """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
                    Return a dictionary which contains config for all input fields.
                    Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
                    Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
                    The type can be a list for selection.

                    Returns: `dict`:
                        - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                        - Value input_fields (`dict`): Contains input fields config:
                            * Key field_name (`string`): Name of a entry-point method's argument
                            * Value field_config (`tuple`):
                                + First value is a string indicate the type of field or a list for selection.
                                + Secound value is a config for type "INT", "STRING" or "FLOAT".
                """
        return {
            "required": {
                "folder_path": ("STRING", {"default": './ComfyUI/input/', "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_by_index"

    # OUTPUT_NODE = False

    CATEGORY = "image"

    def load_by_index(self, folder_path, index):
        def get_file_list(folder_path):
            return os.listdir(folder_path).sort()
        image_path = os.path.join(folder_path, get_file_list(folder_path)[index])
        # Copied from LoadImage
        i = PIL.Image.open(image_path)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadImageInFolder": LoadImageInFolder
}
