import openai
import base64

def encode_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_explanation(diff_img_path):
    image_base64 = encode_image_base64(diff_img_path)
    prompt = (
        "This is a heatmap image showing the differences between two UI layouts. "
        "Please explain what visual changes are highlighted and whether they indicate a layout regression."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ]
    )
    return response['choices'][0]['message']['content']
