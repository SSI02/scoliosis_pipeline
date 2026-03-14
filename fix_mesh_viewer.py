
import os

path = r"e:\scoliosis\pipeline\pipeline\mesh_viewer.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Define the target block to replace
target_block = """        const colArr = new Float32Array(verts.length * 3);
        for (let i = 0; i < verts.length; i++) {
            const v = verts[i][upIdx];
            const t = (v - minUp) / rangeUp;          // 0..1
            colArr[i*3]   = t;                          // R
            colArr[i*3+1] = 0.25;                       // G
            colArr[i*3+2] = 1 - t;                      // B
        }"""

replacement_block = """        const colArr = new Float32Array(verts.length * 3);
        const cVal = 0.8;
        for (let i = 0; i < verts.length; i++) {
            colArr[i*3]   = cVal;
            colArr[i*3+1] = cVal;
            colArr[i*3+2] = cVal;
        }"""

if target_block in content:
    new_content = content.replace(target_block, replacement_block)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully replaced color logic.")
else:
    print("Target block not found. Checking for variations...")
    # Clean up whitespace for checking
    # (Optional: fallback to fuzzy match if needed, but let's see)
    pass
