import json

# Python dictionary
data = {
    "name": "Jac",
    "age": 29,
    "is_student": False,
    "skills": ["Python", "Data Analysis"]
}

# Serialize to JSON string
json_string = json.dumps(data)
print(json_string)
