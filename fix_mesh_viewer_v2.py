
import os

path = r"e:\scoliosis\pipeline\pipeline\mesh_viewer.py"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_loop = False
loop_indent = ""

for i, line in enumerate(lines):
    # Detect the specific loop for color assignment
    if "for (let i = 0; i < verts.length; i++) {" in line and "const colArr" in lines[i-1]:
        in_loop = True
        loop_indent = line[:line.find("for")]
        new_lines.append(line)
        
        # Inject uniform color logic
        new_lines.append(f"{loop_indent}    colArr[i*3]   = 0.8;\n")
        new_lines.append(f"{loop_indent}    colArr[i*3+1] = 0.8;\n")
        new_lines.append(f"{loop_indent}    colArr[i*3+2] = 0.8;\n")
        continue
        
    if in_loop:
        if "}" in line and line.strip() == "}":
            in_loop = False
            new_lines.append(line) # Closing brace
        else:
            # Skip old loop body lines
            pass
    else:
        new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Finished processing mesh_viewer.py")
