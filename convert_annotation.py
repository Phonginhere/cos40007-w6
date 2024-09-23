import os
import pandas as pd

def convert_and_save_yolo_format(df, path):
    class_idx = 0  # Assuming only one class in the dataset

    # Calculate YOLO format using vectorized operations
    x_center = (df['xmin'] + df['xmax']) / 2 / df['width']
    y_center = (df['ymin'] + df['ymax']) / 2 / df['height']
    bbox_width = (df['xmax'] - df['xmin']) / df['width']
    bbox_height = (df['ymax'] - df['ymin']) / df['height']

    # Create YOLO formatted DataFrame
    yolo_df = pd.DataFrame({
        'filename': df['filename'],
        'class_idx': class_idx,
        'x_center': x_center,
        'y_center': y_center,
        'bbox_width': bbox_width,
        'bbox_height': bbox_height
    })

    os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist

    # Group by filename and save each set of annotations
    for filename, group in yolo_df.groupby('filename'):
        file_base = os.path.splitext(filename)[0]
        output_file = os.path.join(path, f"{file_base}.txt")

        # Save YOLO format to text file
        group[['class_idx', 'x_center', 'y_center', 'bbox_width', 'bbox_height']].to_csv(
            output_file, header=False, index=False, sep=' ')

    #print(f"YOLO annotations saved to {path}")

df_train = pd.read_csv('D:/phong/Bounding_boxes/train_labels.csv')
df_test = pd.read_csv('D:/phong/Bounding_boxes/test_labels.csv')

path1 = 'D:/phong/dataset/labels/train'
path2 = 'D:/phong/dataset/labels/test'

convert_and_save_yolo_format(df_train, path1)
convert_and_save_yolo_format(df_test, path2)
