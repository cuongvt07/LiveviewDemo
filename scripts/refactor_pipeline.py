import os
import re

md_path = r'd:\Design System\Pod mockup architecture.md'
base_dir = r'd:\Design System\mug-mockup-service\app\pipeline'

# Read MD file
with open(md_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

current_file = None
in_code = False
code_lines = []
files_created = 0

for line in lines:
    # Match headers
    header_match = re.search(r'#+\s.*?`(.+\.py)`', line)
    if header_match:
        current_file = header_match.group(1).strip()
        continue
    
    if line.startswith('```python'):
        in_code = True
        code_lines = []
        continue
    
    if line.startswith('```') and in_code:
        in_code = False
        if current_file:
            out_path = os.path.join(base_dir, current_file)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as mf:
                mf.writelines(code_lines)
            print(f"Created {current_file}")
            files_created += 1
            current_file = None
        continue
    
    if in_code:
        code_lines.append(line)

# Create empty __init__.py files
for d in ['shared', 'mugs', 'clothes']:
    open(os.path.join(base_dir, d, '__init__.py'), 'w').close()

print(f"Done extracting {files_created} files.")
