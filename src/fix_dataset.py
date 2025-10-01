# # fix_dataset.py
# import os
#
# # Fix the dataset.py file
# dataset_file = 'dataset.py'
#
# if not os.path.exists(dataset_file):
#     print("dataset.py not found!")
#     exit(1)
#
# with open(dataset_file, 'r') as f:
#     content = f.read()
#
# # Check if already fixed
# if 'elif task_name == "rte":' in content and 'sentence1' in content:
#     print("dataset.py already appears to be fixed!")
#     exit(0)
#
# # Fix the RTE handling
# old_pattern = '''elif task_name in ["mnli", "rte"]:
#                 return sample["premise"], sample["hypothesis"]'''
#
# new_pattern = '''elif task_name == "mnli":
#                 return sample["premise"], sample["hypothesis"]
#             elif task_name == "rte":
#                 return sample["sentence1"], sample["sentence2"]'''
#
# if old_pattern.replace(' ', '').replace('\n', '') in content.replace(' ', '').replace('\n', ''):
#     content = content.replace(old_pattern, new_pattern)
#
#     with open(dataset_file, 'w') as f:
#         f.write(content)
#
#     print("✅ Fixed dataset.py - RTE now uses sentence1/sentence2")
# else:
#     print("⚠️  Could not find the exact pattern to fix. Please manually edit src/dataset.py")
#     print("Change the _extract_texts method to handle RTE separately:")
#     print('    elif task_name == "rte":')
#     print('        return sample["sentence1"], sample["sentence2"]')

# verify_fix.py
import os

dataset_file = 'dataset.py'

with open(dataset_file, 'r') as f:
    content = f.read()

print("=== Checking _extract_texts method ===")

# Find the _extract_texts method
lines = content.split('\n')
in_extract_texts = False
extract_texts_lines = []

for line in lines:
    if 'def _extract_texts(self' in line:
        in_extract_texts = True
    elif in_extract_texts and line.strip().startswith('def ') and '_extract_texts' not in line:
        break

    if in_extract_texts:
        extract_texts_lines.append(line)

print("Found _extract_texts method:")
for i, line in enumerate(extract_texts_lines):
    print(f"{i + 1:2d}: {line}")

# Check specifically for RTE handling
rte_lines = [line for line in extract_texts_lines if 'rte' in line.lower()]
print(f"\nRTE-related lines: {len(rte_lines)}")
for line in rte_lines:
    print(f"  {line.strip()}")

# Check if the problematic pattern exists
bad_pattern = 'task_name in ["mnli", "rte"]'
if bad_pattern in content:
    print(f"\n❌ PROBLEM: Found bad pattern: {bad_pattern}")
    print("The fix didn't work properly!")
else:
    print(f"\n✅ Good: Bad pattern not found")