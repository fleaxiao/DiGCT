import os
from PIL import Image, ImageDraw
import shutil

input_folder = "./data"
output_folder = "./dataset"

angle_step = 30

surface_start = 378
surface_end = 2526
side_start = 132
side_end = 2772

surface_center = surface_start + (surface_end - surface_start) // 2
surface_radius = (surface_end - surface_start) // 2
side_length = side_end - side_start

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    ## Surface images
    if filename.startswith("surface") and filename.endswith(".png") and not filename.endswith("_0.png"):
        input_path = os.path.join(input_folder, filename)
        
        with Image.open(input_path) as img:
            img = img.convert("RGBA")

            width, height = img.size
            left = surface_center - surface_radius
            top = height // 2 - surface_radius
            right = surface_center + surface_radius
            bottom = height // 2 + surface_radius
            
            # transparent circle crop
            cropped_img = img.crop((left, top, right, bottom))
            
            mask = Image.new("L", (right - left, bottom - top), 0)
            draw = ImageDraw.Draw(mask)

            draw.ellipse((0, 0, right - left, bottom - top), fill=255)
            cropped_img.putalpha(mask)
            
            # rotation and save the images
            for angle in range(0, 360, angle_step):
                rotated_img = cropped_img.rotate(angle, expand=False)

                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_rotated_{angle}{ext}"
                output_path = os.path.join(output_folder, output_filename)
                rotated_img.save(output_path)
        
        ## Side images
        if filename.startswith("side") and filename.endswith(".png") and not filename.endswith("_0.png"):
            input_path = os.path.join(input_folder, filename)

            with Image.open(input_path) as img:
                img = img.convert("RGBA")

                # Extract pixels from side_start to side_end along the center line
                width, height = img.size
                center_line = height // 2
                pixels = []
                for x in range(side_start, side_end):
                    pixel = img.getpixel((x, center_line))
                    pixels.append(pixel)

                pixels_extend = pixels.copy()
                reversed_pixels = list(reversed(pixels))
                pixels_extend.extend(reversed_pixels)
                pixels_extend.extend(pixels)

                for angle in range(0, 360, angle_step):
                    start_pixel = int(2* angle * side_length / 360)
                    pixels_rotated = pixels_extend[start_pixel:start_pixel + side_length]

                    img_rotated = Image.new("RGBA", (len(pixels_rotated), 1), (0, 0, 0, 0))
                    img_rotated.putdata(pixels_rotated)

                    name, ext = os.path.splitext(filename)
                    output_filename = f"{name}_rotated_{angle}{ext}"
                    output_path = os.path.join(output_folder, output_filename)
                    img_rotated.save(output_path)

            ## Range csv
            csv_source_path = os.path.join(input_folder, "T_range.csv")
            csv_destination_path = os.path.join(output_folder, "T_range.csv")
            shutil.copy(csv_source_path, csv_destination_path)