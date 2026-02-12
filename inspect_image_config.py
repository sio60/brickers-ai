import sys
from google.genai import types as genai_types

try:
    print("Fields in ImageConfig:")
    if hasattr(genai_types, 'ImageConfig'):
         print(genai_types.ImageConfig.model_fields.keys())
    else:
         print("ImageConfig not found in genai_types")

except Exception as e:
    print(f"Error: {e}")
