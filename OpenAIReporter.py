import openai
import base64

# Initialize OpenAI client
client = openai.OpenAI(api_key="")


# Helper to encode image in base64
def encode_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# Paths to your baseline and diff images
baseline_img_path = "dataset/test_images/counter_app_clean.png"
diff_img_path = "dataset/test_images/diff_login.png"

# Encode images
baseline_img_b64 = encode_image_base64(baseline_img_path)
diff_img_b64 = encode_image_base64(diff_img_path)

# Send request to GPT-4o
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert UI tester who identifies layout bugs."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "These two images show a baseline layout and a diff image with red boxes highlighting issues. "
                        "Compare them and generate a UI layout regression bug report with these sections:\n"
                        "- Baseline Layout\n- Regressed Layout\n"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{baseline_img_b64}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{diff_img_b64}"}
                }
            ]
        }
    ],
    max_tokens=1000
)

# Print the result
print("\n📋 Layout Regression Bug Report:\n")
print(response.choices[0].message.content)
