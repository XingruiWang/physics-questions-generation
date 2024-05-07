from convert_video import convert_to_gif
import json
import os
import random
from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    question_file = "/home/xingrui/physics_questions_generation/output/debug_200_questions.json"
    folder_path = "/home/xingrui/physics_questions_generation/data/output"
    output_gif_path = "."
    
    with open(question_file, "r") as f:
        questions = json.load(f)
    questions = questions['questions']
    idx = 0
    while idx < 10:
        rand_questions = random.sample(questions, 1)
        question = rand_questions[0]
        question, answer, image_filename, template_filename = question['question'], question['answer'], question['image_filename'], question['template_filename']

        print(f"\nQuestion {idx}: {question}; Answer: {answer}; Image: {image_filename}; Template: {template_filename}")
        # os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
        gif_path = convert_to_gif(folder_path, output_gif_path, image_filename, anno = {'question': question, 'answer': answer})
        idx += 1