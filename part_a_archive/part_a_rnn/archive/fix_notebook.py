#!/usr/bin/env python3
"""
Script to fix ultimate_complete_pipeline.ipynb:
1. Reorder cells by section number (1-28)
2. Add missing Section 3 header
3. Remove all emojis from all cells
"""

import json
import re
from pathlib import Path

def remove_emojis(text):
    """Remove all emoji characters from text"""
    # Comprehensive emoji pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def extract_section_number(source_text):
    """Extract section number from markdown header"""
    # Look for pattern like "Section 1:" or "Section 1 "
    match = re.search(r'Section\s+(\d+)', source_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def fix_notebook(notebook_path):
    """Fix the notebook: reorder, add Section 3, remove emojis"""

    print(f"Loading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Total cells: {len(cells)}")

    # Step 1: Categorize cells
    title_cell = None
    section_cells = {}  # section_num -> list of cells belonging to that section
    current_section = None

    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

            # Check if it's the title cell
            if i == 0 and source.strip().startswith('# Emotion Detection Pipeline'):
                title_cell = cell
                continue

            # Check if it's a section header
            section_num = extract_section_number(source)

            # Special handling for Text Preprocessor (missing Section 3)
            if 'Text Preprocessor with Statistics' in source and '## Text Preprocessor' in source:
                # This is the missing Section 3
                section_num = 3
                # Fix the header to include Section 3
                source = source.replace('## Text Preprocessor with Statistics',
                                      '## Section 3: Advanced Text Preprocessor with Statistics')
                if isinstance(cell['source'], list):
                    cell['source'] = [source]
                else:
                    cell['source'] = source
                print(f"Fixed Section 3 header at cell {i}")

            if section_num is not None:
                current_section = section_num
                if current_section not in section_cells:
                    section_cells[current_section] = []
                section_cells[current_section].append(cell)
            elif current_section is not None:
                section_cells[current_section].append(cell)
            else:
                # Cell before any section (shouldn't happen after title)
                if current_section is None:
                    current_section = 0
                    if current_section not in section_cells:
                        section_cells[current_section] = []
                section_cells[current_section].append(cell)
        else:
            # Code cells belong to current section
            if current_section is not None:
                section_cells[current_section].append(cell)
            else:
                # Code cell before any section
                if 0 not in section_cells:
                    section_cells[0] = []
                section_cells[0].append(cell)

    print(f"\nSections found: {sorted([s for s in section_cells.keys() if s > 0])}")

    # Step 2: Remove emojis from all cells
    print("\nRemoving emojis from all cells...")
    emoji_count = 0

    def clean_cell_source(cell):
        nonlocal emoji_count
        if isinstance(cell['source'], list):
            cleaned_source = []
            for line in cell['source']:
                original_line = line
                cleaned_line = remove_emojis(line)
                if original_line != cleaned_line:
                    emoji_count += 1
                cleaned_source.append(cleaned_line)
            cell['source'] = cleaned_source
        else:
            original = cell['source']
            cleaned = remove_emojis(original)
            if original != cleaned:
                emoji_count += 1
            cell['source'] = cleaned

    # Clean title cell
    if title_cell:
        clean_cell_source(title_cell)

    # Clean all section cells
    for section_num, cells_list in section_cells.items():
        for cell in cells_list:
            clean_cell_source(cell)

    print(f"Removed emojis from {emoji_count} cell lines/sources")

    # Step 3: Rebuild notebook in correct order
    print("\nReordering cells...")
    new_cells = []

    # Add title cell first
    if title_cell:
        new_cells.append(title_cell)

    # Add cells from section 0 (setup before Section 1)
    if 0 in section_cells:
        new_cells.extend(section_cells[0])

    # Add sections 1-28 in order
    for section_num in range(1, 29):
        if section_num in section_cells:
            new_cells.extend(section_cells[section_num])
            print(f"  Added Section {section_num}: {len(section_cells[section_num])} cells")
        else:
            print(f"  WARNING: Section {section_num} not found!")

    # Update notebook with new cell order
    nb['cells'] = new_cells

    # Step 4: Save fixed notebook
    output_path = notebook_path.replace('.ipynb', '_fixed.ipynb')
    print(f"\nSaving fixed notebook to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\nSuccess! Fixed notebook saved with {len(new_cells)} cells")
    print(f"Original: {notebook_path}")
    print(f"Fixed: {output_path}")

    return output_path

if __name__ == '__main__':
    notebook_path = '/home/lab/rabanof/projects/Emotion_Detection_DL/notebooks/ultimate_complete_pipeline.ipynb'
    fixed_path = fix_notebook(notebook_path)

    # Verify the fix
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    with open(fixed_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"\nTotal cells in fixed notebook: {len(nb['cells'])}")
    print("\nSection order:")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'Section' in source:
                first_line = source.split('\n')[0].strip()
                print(f"  Cell {i}: {first_line[:80]}")

    print("\n" + "="*60)
    print("Done!")
