import json

class JSONArrayToStringList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"json_array": ("STRING",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("texts",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "utils"

    def run(self, json_array):
        data = json.loads(json_array)
        if not isinstance(data, list):
            raise ValueError("Input is not a JSON array")
        return (data,)

NODE_CLASS_MAPPINGS = {"JSONArrayToStringList": JSONArrayToStringList}
NODE_DISPLAY_NAME_MAPPINGS = {"JSONArrayToStringList": "JSON Array → String List"}
