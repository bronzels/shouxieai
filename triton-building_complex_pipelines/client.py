import time

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *


def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    prompt = "Pikachu with a hat, 4k, 3d render"
    text_obj = np.array([prompt], dtype="object").reshape((-1, 1))

    input_text = httpclient.InferInput(
        "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
    )
    input_text.set_data_from_numpy(text_obj)

    output_img = httpclient.InferRequestedOutput("generated_image")

    query_response = client.infer(
        model_name="pipeline", inputs=[input_text], outputs=[output_img]
    )

    image = query_response.as_numpy("generated_image")
    im = Image.fromarray(np.squeeze(image.astype(np.uint8)))
    im.save("generated_image2.jpg")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print("Time taken:", end - start)
