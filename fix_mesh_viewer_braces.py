
import os

path = r"e:\scoliosis\pipeline\pipeline\mesh_viewer.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix over-escaped braces
new_content = content.replace("{{{{", "{{").replace("}}}}", "}}")

with open(path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Fixed braces in mesh_viewer.py")
