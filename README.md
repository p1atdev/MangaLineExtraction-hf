# MangaLineExtraction-hf

A huggingface transformers compatible implementation of MangaLineExtraction (https://github.com/ljsabc/MangaLineExtraction_PyTorch).


The converted weights are avaiable on [🤗 HuggingFace](https://huggingface.co/p1atdev/MangaLineExtraction-hf).

## Example usage with transformers

```py
from PIL import Image
import torch

from transformers import AutoModel, AutoImageProcessor

REPO_NAME = "p1atdev/MangaLineExtraction-hf"

model = AutoModel.from_pretrained(REPO_NAME, trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained(REPO_NAME, trust_remote_code=True)

image = Image.open("./sample.jpg")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(inputs.pixel_values)

line_image = Image.fromarray(outputs.pixel_values[0].numpy().astype("uint8"), mode="L")
line_image.save("./line_image.png")
```

## Acknowledgements

We extend our gratitude to the authors of the  [Deep Extraction of Manga Structural Lines](https://www.cse.cuhk.edu.hk/~ttwong/papers/linelearn/linelearn.html) and the contributors to the  [MangaLineExtraction_PyTorch](https://github.com/ljsabc/MangaLineExtraction_PyTorch) for their pioneering work that served as the foundation for our adaptation. Our thanks also go to HuggingFace for developing [transformers](https://github.com/huggingface/transformers), enabling us to enhance this project further.

For detailed information, please refer to:
- https://www.cse.cuhk.edu.hk/~ttwong/papers/linelearn/linelearn.html
- https://github.com/huggingface/transformers
