import os
import csv

def save_png_filenames_and_classes_to_csv(folder_path, csv_file_path):
    files = os.listdir(folder_path)

    png_files = [file for file in files if file.endswith('.tif')]
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for png in png_files:
            parts = png.split('-')
            if len(parts) > 2:
                class_name = parts[1] 
            else:
                class_name = 'unknown'
            
            writer.writerow([png, class_name])

folder_train_path = './us8k_train' # your path
csv_train_file_path = './us8k_train.csv' # your path

folder_valid_path = './us8k_valid'  # your path
csv_valid_file_path = './us8k_valid.csv'  # your path

save_png_filenames_and_classes_to_csv(folder_train_path, csv_train_file_path)
save_png_filenames_and_classes_to_csv(folder_valid_path, csv_valid_file_path)
