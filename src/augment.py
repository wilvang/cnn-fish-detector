from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import numpy as np

# Base input and output directories
input_folder = 'C:/Users/ellin/OneDrive/Skrivebord/Skole/fishproject/AnadromSmall(Fish Dataset)/LiteEksempel/Fish'
base_output_folder = 'C:/Users/ellin/OneDrive/Skrivebord/Skole/fishproject/AnadromSmall(Fish Dataset)/LiteEksempel/Processed'

# Define different preprocessing categories and their specific output folders
preprocessing_folders = {
    "resized": os.path.join(base_output_folder, "Resized"),
    "grayscale": os.path.join(base_output_folder, "Grayscale"),
    "rotated": os.path.join(base_output_folder, "Rotated"),
    "flipped": os.path.join(base_output_folder, "Flipped"),
    "color_spaces": os.path.join(base_output_folder, "ColorSpaces"),
    "lighting": os.path.join(base_output_folder, "Lighting"),
    "contrast": os.path.join(base_output_folder, "Contrast"),
    "filters": os.path.join(base_output_folder, "Filters"),
    "water_effects": os.path.join(base_output_folder, "WaterEffects")
}

# Create output folders if they don't exist
for folder in preprocessing_folders.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to simulate underwater/muddy water effect
def apply_underwater_effect(image, turbidity=0.5):
    """Apply underwater effect to an image with variable turbidity (0-1)"""
    img_array = np.array(image).astype(float)
    
    # Reduce red channel (water absorbs red light first)
    red_reduction = 1.0 - (0.6 * turbidity)
    img_array[:,:,0] = img_array[:,:,0] * red_reduction
    
    # Add blue/green tint (more blue for higher turbidity)
    blue_tint = 20 * turbidity
    green_tint = 10 * turbidity
    
    img_array[:,:,2] = np.minimum(img_array[:,:,2] + blue_tint, 255)
    img_array[:,:,1] = np.minimum(img_array[:,:,1] + green_tint, 255)
    
    # Add blur to simulate particles in water
    result = Image.fromarray(img_array.astype(np.uint8))
    blur_radius = 1 + turbidity * 2
    result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return result

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        base, ext = os.path.splitext(filename)
        
        # 1. Resizing variations
        sizes = [(224, 224), (299, 299), (512, 512)]
        for size in sizes:
            resized = img.resize(size)
            resized.save(os.path.join(preprocessing_folders["resized"], f"{base}_resized_{size[0]}x{size[1]}{ext}"))
        
        # 2. Grayscale
        grayscale = ImageOps.grayscale(img)
        grayscale.save(os.path.join(preprocessing_folders["grayscale"], f"{base}_grayscale{ext}"))
        
        # 3. Rotations
        angles = [90, 180, 270]
        for angle in angles:
            rotated = img.rotate(angle)
            rotated.save(os.path.join(preprocessing_folders["rotated"], f"{base}_rotated_{angle}{ext}"))
        
        # 4. Flips
        flipped_h = ImageOps.mirror(img)
        flipped_h.save(os.path.join(preprocessing_folders["flipped"], f"{base}_flipped_h{ext}"))
        flipped_v = ImageOps.flip(img)
        flipped_v.save(os.path.join(preprocessing_folders["flipped"], f"{base}_flipped_v{ext}"))
        
        # 5. Color space variations (HSV-like manipulation)
        if img.mode == "RGB":
            # Convert to HSV-like by manipulating channels
            img_array = np.array(img)
            hue_shift = img_array.copy()
            hue_shift[:,:,[0,1,2]] = img_array[:,:,[1,2,0]]  # Shift hue channels
            hue_image = Image.fromarray(hue_shift)
            hue_image.save(os.path.join(preprocessing_folders["color_spaces"], f"{base}_hue_shift{ext}"))
        
        # 6. Lighting variations
        brightness_factors = [0.7, 1.3, 1.5]
        for factor in brightness_factors:
            enhancer = ImageEnhance.Brightness(img)
            brightened = enhancer.enhance(factor)
            brightened.save(os.path.join(preprocessing_folders["lighting"], f"{base}_brightness_{int(factor*100)}{ext}"))
        
        # 7. Contrast variations
        contrast_factors = [0.5, 1.5, 2.0]
        for factor in contrast_factors:
            enhancer = ImageEnhance.Contrast(img)
            contrasted = enhancer.enhance(factor)
            contrasted.save(os.path.join(preprocessing_folders["contrast"], f"{base}_contrast_{int(factor*100)}{ext}"))
        
        # 8. Various filters
        # Gaussian blur
        img.filter(ImageFilter.GaussianBlur(radius=2)).save(
            os.path.join(preprocessing_folders["filters"], f"{base}_gaussian_blur{ext}"))
        
        # Edge enhancement
        img.filter(ImageFilter.EDGE_ENHANCE).save(
            os.path.join(preprocessing_folders["filters"], f"{base}_edge_enhance{ext}"))
        
        # Sharpen
        img.filter(ImageFilter.SHARPEN).save(
            os.path.join(preprocessing_folders["filters"], f"{base}_sharpen{ext}"))
        
        # 9. Water effects - simulating underwater/muddy conditions
        turbidity_levels = [0.2, 0.5, 0.8]
        for level in turbidity_levels:
            underwater = apply_underwater_effect(img, turbidity=level)
            underwater.save(os.path.join(
                preprocessing_folders["water_effects"], 
                f"{base}_underwater_{int(level*100)}{ext}"))

print("Image preprocessing complete. Check output folders for results.")