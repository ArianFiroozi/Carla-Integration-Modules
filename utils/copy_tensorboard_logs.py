import os
import shutil
import argparse

def copy_tensorboard_logs(src_dir, dst_dir):
    """
    Traverses the src_dir, finds all tensorboard log files (matching *tfevents*),
    and copies them to dst_dir while preserving the directory structure.
    """
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    
    print(f"Scanning '{src_dir}' for TensorBoard logs...")
    print(f"Preserved structure will be copied to '{dst_dir}'\n")
    
    copied_count = 0
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Tensorboard event files typically contain 'tfevents' in their name
            if "tfevents" in file:
                src_file_path = os.path.join(root, file)
                
                # Calculate relative path to maintain directory structure
                rel_path = os.path.relpath(src_file_path, src_dir)
                dst_file_path = os.path.join(dst_dir, rel_path)
                
                # Create destination directory if it doesn't exist
                dst_file_dir = os.path.dirname(dst_file_path)
                os.makedirs(dst_file_dir, exist_ok=True)
                
                # Copy the file
                print(f"Copying: {rel_path}")
                shutil.copy2(src_file_path, dst_file_path)
                copied_count += 1
                
    print(f"\nDone! Copied {copied_count} TensorBoard log files successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy TensorBoard log files from an experiments directory while preserving folder structure.")
    parser.add_argument("--src", default="experiments", help="Path to the source experiments directory (default: 'experiments')")
    parser.add_argument("--dst", default="experiments_tb_only", help="Path to the destination directory (default: 'experiments_tb_only')")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.src):
        print(f"Error: Source directory '{args.src}' does not exist.")
    else:
        copy_tensorboard_logs(args.src, args.dst)
