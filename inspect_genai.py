import sys
import os
from google.genai import types as genai_types

print("Fields in GenerateContentConfig:")
try:
    # Try pydantic fields
    print(genai_types.GenerateContentConfig.model_fields.keys())
except:
    print(dir(genai_types.GenerateContentConfig))
