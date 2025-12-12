import os
import glob
import sys
import argparse

def rename_images(folder_path, prefix="taag", start_number=1, dry_run=False):

    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []

    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_files.sort()
    
    if not image_files:
        print(f"No image found : {folder_path}")
        return

    print(f"--- {len(image_files)} images found. Start renaming... ---")

    counter = start_number
    for old_path in image_files:
        root, ext = os.path.splitext(old_path)
  
        new_name = f"{prefix}_{counter:04d}{ext.lower()}"
        new_path = os.path.join(folder_path, new_name)

        action = f"Rename : {os.path.basename(old_path)} -> {new_name}"
        
        if dry_run:
            print(f"[SIMULATION] {action}")
        else:
            try:
                os.rename(old_path, new_path)
                print(action)
            except Exception as e:
                print(f"Error : impossible to rename {os.path.basename(old_path)}: {e}")
        
        counter += 1

    if dry_run:
        print("\n*** No file renamed ***")
    else:
        print("\n*** Renaming ended. ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Renomme séquentiellement toutes les images d'un dossier donné.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("folder_path", type=str, 
                        help="Path to the folder.")
    
    parser.add_argument("-p", "--prefix", type=str, default="taag",
                        help="Prefix (default: 'taag').")
                        
    parser.add_argument("-s", "--start", type=int, default=1,
                        help="Starting number of the sequence (Default: 1, ex: taag_0001).")
                        
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    
    rename_images(args.folder_path, args.prefix, args.start, args.dry_run)