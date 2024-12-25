import sys
from PIL import Image

def convert_background(input_path, output_path, to_white=True):
    img = Image.open(input_path)
    
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    if to_white:
        new_img = Image.new('RGB', img.size, (255, 255, 255))
        new_img.paste(img, (0, 0), img)
        new_img.save(output_path)
    else:
        datas = img.getdata()
        new_data = []
        for item in datas:
            if item[0] > 200 and item[1] > 200 and item[2] > 200:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        img.putdata(new_data)
        img.save(output_path, "PNG")

input_path = sys.argv[1]
output_path = sys.argv[2]

convert_background(input_path, output_path, to_white=False)