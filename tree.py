import os

print("full_structure")

def print_tree(dir_path, prefix=""):
    # Faltu folders jo hide karne hain
    ignore_dirs = {'.git', '__pycache__', 'venv', 'env', '.venv', '.agents', '.codex', 'node_modules', 'outputs'}
    # Faltu files jo hide karni hain
    ignore_exts = {'.pyc', '.pyo', '.png', '.jpg', '.jpeg', '.ico', '.pdf', '.zip', '.exe'}

    files_and_dirs = os.listdir(dir_path)
    
    # Folders aur files ko alag karna taaki folders upar aayein (--dirsfirst)
    dirs = [d for d in files_and_dirs if os.path.isdir(os.path.join(dir_path, d)) and d not in ignore_dirs]
    files = [f for f in files_and_dirs if os.path.isfile(os.path.join(dir_path, f)) and not any(f.endswith(ext) for ext in ignore_exts)]
    
    # Alphabetical order mein sort karna
    dirs.sort()
    files.sort()
    
    # Folders pehle, phir files
    contents = dirs + files
    
    tree_str = ""
    for i, item in enumerate(contents):
        # Check karna ki yeh last item hai ya nahi (taaki └── lag sake)
        is_last = (i == len(contents) - 1)
        connector = "└── " if is_last else "├── "
        
        item_path = os.path.join(dir_path, item)
        
        if os.path.isdir(item_path):
            tree_str += f"{prefix}{connector}{item}/\n"
            extension = "    " if is_last else "│   "
            tree_str += print_tree(item_path, prefix + extension)
        else:
            # Script khud ki output file ko hide karne ke liye
            if item not in ["tree_maker.py", "full_structure.txt"]:
                tree_str += f"{prefix}{connector}{item}\n"
            
    return tree_str

if __name__ == "__main__":
    # Root folder ka naam nikalna
    root_name = os.path.basename(os.path.abspath('.'))
    
    # Tree banana shuru karna
    tree_output = f"{root_name}/\n"
    tree_output += "│\n"
    tree_output += print_tree('.')
    
    # Output ko text file mein save karna
    with open("full_structure.txt", "w", encoding="utf-8") as f:
        f.write(tree_output)
        
    print("✅ Success! Ekdum perfect Tree design 'full_structure.txt' mein save ho gaya hai.")