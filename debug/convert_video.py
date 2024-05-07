from PIL import Image, ImageDraw, ImageFont

import os
from tqdm import tqdm
# Define the folder where your images are located
# /home/angtian/xingrui/superclevr2kubric/output/physics_super_clevr_c1/super_clever_0

def wrap_text(text, line_length):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if len(' '.join(current_line + [word])) > line_length:
            lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def overlay_text_on_image(image, question, answer):
    """
    Function to overlay text (question and answer) on an image.

    Args:
    image (PIL.Image.Image): The image on which to overlay text.
    question (str): The question text to overlay on the image.
    answer (str): The answer text to overlay on the image.

    Returns:
    PIL.Image.Image: The new image with text overlaid.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # You can change this to a specific font and size if available
    # Scale the font size up by 10 times using a multiplier
    font_size_multiplier = 5
    scaled_font = ImageFont.load_default().font_variant(size=font.size * font_size_multiplier)


    # Wrap text for question and answer
    question_lines = wrap_text(f"Question: {question}", 50)
    answer_lines = wrap_text(f"Answer: {answer}", 50)

    # Calculate initial y-offset for text placement
    y_offset = 10

    # Draw the wrapped question text
    for line in question_lines:
        draw.text((10, y_offset), line, font=scaled_font, fill="white")
        y_offset += 50  # Increment y-offset for the next line of text

    # Draw the wrapped answer text
    for line in answer_lines:
        draw.text((10, y_offset), line, font=scaled_font, fill="white")
        y_offset += 50  # Increment y-offset for the next line of text

    return image

def convert_to_gif(folder_path, output_gif_path, image_filename, anno = None):
    folder_path = os.path.join(folder_path, image_filename)
    output_gif_path = os.path.join(output_gif_path, f'{image_filename}.gif')
    # output_gif_path = '/home/angtian/xingrui/superclevr2kubric/output/gif_multi/c0/super_clever_{}.gif'.format(idx)
    
    # List to store images
    images = []

    # Load each image
    for i in tqdm(range(50)):  # Assuming the images are numbered from 00000 to 00060
        filename = f'rgba_{i:05d}.png'
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            print(f"File {filename} not found.")
        else:
            img = Image.open(file_path)

            if anno:
                question = anno['question']
                answer = anno['answer']

                img = overlay_text_on_image(img, question, answer)
            images.append(img)
            

    # Save the images as a GIF
    if images:
        # images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
        images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
        print(f"GIF saved at {output_gif_path}")
    else:
        print("No images were loaded to create a GIF.")
    return output_gif_path
if __name__ == "__main__":
    for idx in tqdm(range(0,1)):
        folder_path = '/home/angtian/xingrui/superclevr2kubric/output/physics_super_clevr_c0/super_clever_{}/'.format(idx)
        output_gif_path = '/home/angtian/xingrui/superclevr2kubric/output/gif_multi/c0/super_clever_{}.gif'.format(idx)
        os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
        # List to store images
        images = []

        # Load each image
        for i in tqdm(range(50)):  # Assuming the images are numbered from 00000 to 00060
            filename = f'rgba_{i:05d}.png'
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                print(f"File {filename} not found.")
            else:
                img = Image.open(file_path)
                images.append(img)
                

        # Save the images as a GIF
        if images:
            # images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
            images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
            print(f"GIF saved at {output_gif_path}")
        else:
            print("No images were loaded to create a GIF.")