import cv2
import numpy as np

# Path to the sample images
orig_path = 'Orig_images/0000606_orig.png'
model_iyi_path = 'Model_IYI/0000606_blur_output.png'
model_kotu_path = 'Model_KOTU/0000606_blur_output.png'
restormer_orig_path = 'RestormerOrig8Epoch/0000606_blur.png'
restormer_project_path = 'RestormerProject/0000606_blur.png'
blurred_path = 'Blurred_Images/0000606_blur.png'


# Read the sample images
orig_img = cv2.imread(orig_path)
model_iyi_img = cv2.imread(model_iyi_path)
model_kotu_img = cv2.imread(model_kotu_path)
restormer_orig_img = cv2.imread(restormer_orig_path)
restormer_project_img = cv2.imread(restormer_project_path)
blurred_img = cv2.imread(blurred_path)

# Resize the images to a fixed size for the grid
fixed_size = (200, 200)
orig_img = cv2.resize(orig_img, fixed_size)
model_iyi_img = cv2.resize(model_iyi_img, fixed_size)
model_kotu_img = cv2.resize(model_kotu_img, fixed_size)
restormer_orig_img = cv2.resize(restormer_orig_img, fixed_size)
restormer_project_img = cv2.resize(restormer_project_img, fixed_size)
blurred_img = cv2.resize(blurred_img, fixed_size)

# Create an empty grid image
grid_width = 3 * fixed_size[0]
grid_height = 2 * fixed_size[1] + 50
table_height = 50

grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

# Write descriptions at the top of the grid image
descriptions = ['Blurred', 'Model IYI', 'Model Kotu', 'Restormer Orig', 'Restormer Project', 'Orig']
text_color = (255, 255, 255)  # White text color
text_scale = 0.8
text_thickness = 2
text_padding = 10

for i, desc in enumerate(descriptions):
    text_size = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
    text_x = (i % 3) * fixed_size[0] + (fixed_size[0] - text_size[0]) // 2
    text_y = table_height // 2 + text_size[1] // 2
    cv2.putText(grid_img, desc, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness, cv2.LINE_AA)

# Fill the grid with the sample images
grid_img[table_height:table_height+fixed_size[1], 0:fixed_size[0]] = blurred_img
grid_img[table_height:table_height+fixed_size[1], fixed_size[0]:2*fixed_size[0]] = model_iyi_img
grid_img[table_height:table_height+fixed_size[1], 2*fixed_size[0]:] = model_kotu_img
grid_img[table_height+fixed_size[1]:, 0:fixed_size[0]] = restormer_orig_img
grid_img[table_height+fixed_size[1]:, fixed_size[0]:2*fixed_size[0]] = restormer_project_img
grid_img[table_height+fixed_size[1]:, 2*fixed_size[0]:] = orig_img

# Display the grid image
cv2.imshow('Sample Images', grid_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the grid image
cv2.imwrite('sample_images_table.png', grid_img)

