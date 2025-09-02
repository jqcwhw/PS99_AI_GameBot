# Serialization to a File

# To write JSON data directly to a file, use json.dump().

# Write JSON to a file
with open("data.json", "w") as file:
    json.dump(data, file)

# Custom Serialization, You can customize serialization 
  using parameters like indent, separators, and sort_keys.

# Pretty-print JSON with indentation
json_string = json.dumps(data, indent=4, sort_keys=True)
print(json_string)

# Handling Non-Serializable Objects - If your object isn't JSON 
  serializable (e.g., a custom class), you can define a custom encoder


class CustomObject:
    def __init__(self, name, value):
        self.name = name
        self.value = value

obj = CustomObject("example", 42)

# Custom encoder
def custom_encoder(o):
    if isinstance(o, CustomObject):
        return {"name": o.name, "value": o.value}
    raise TypeError("Object not serializable")

json_string = json.dumps(obj, default=custom_encoder)
print(json_string)