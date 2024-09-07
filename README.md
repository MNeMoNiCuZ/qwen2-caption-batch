# Qwen2 Caption Batch
This tool uses the VLM [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) from Qwen to caption images in an input folder. Thanks to their team for training this great model.

It's a very fast and fairly robust captioning model that can produce good outputs in 3 different levels of detail.

## Requirements
* Python 3.10 or 3.11.
  * It's been tested with 3.10 and 3.11

* Cuda 12.1.
  * It may work with other versions. Untested.
 
* GPU with 24gb VRAM
  * It gets out of memory on a 3090 unless images are resized to ~1024x1024 (included in the script)

## Setup
1. Create a virtual environment. Use the included `venv_create.bat` to automatically create it. Use python 3.10 or above.
2. Install the libraries in requirements.txt. `pip install -r requirements.txt`. This is done by step 1 when asked if you use `venv_create`.
3. Install [Pytorch for your version of CUDA](https://pytorch.org/). It's only been tested with version 12.1 but may work with others.
4. Open `batch.py` in a text editor and edit any settings you want.


## How to use
1. Activate the virtual environment. If you installed with `venv_create.bat`, you can run `venv_activate.bat`.
2. Run `python batch.py` from the virtual environment.

This runs captioning on all images in the /input/-folder.

## Example prompts
You can edit `batch.py` in a text editor and edit the DEFAULT_PROMPT value to change the captioning prompt.

![IndependenceDaze](https://github.com/user-attachments/assets/dbfe27af-e475-4dd3-9f58-a1adf16eab9b)


Here are some example prompts and their captions:
### Quick caption
In one short sentence, caption the key aspects of this image
> A man in sunglasses holds coffee and doughnuts against an apocalyptic cityscape with a UFO overhead; text reads "Independence Daze: The battle for Earth has begun... after brunch."
---
### Medium caption
In one medium sentence, caption the key aspects of this image
> The poster for "Independence Daze" features an action-packed scene with a man holding coffee and doughnuts against a backdrop of a cityscape under threat from extraterrestrial activity; it humorously suggests that Earth's battle has begun after brunch.
---
### Long caption
Describe all details of this image
> The image is a movie poster for the film "Independence Daze." The title, written in bold letters at the top with an American flag motif behind it and stars scattered around, suggests themes related to independence or freedom.  In the foreground, there's a man wearing sunglasses, holding a blue coffee cup in one hand and a large doughnut covered in sugar sprinkles in his other hand. He appears relaxed yet determined as he stands against a backdrop that includes a cityscape with tall buildings under dramatic lighting conditionsclouds illuminated by what seems like sunlight filtering through from above.  Above him floats a flying saucer-like object emitting light rays downward towards Earth, indicating some form of extraterrestrial activity or intervention. This adds elements typical of science fiction movies involving alien encounters or space battles.  At the bottom part of the poster reads: THE BATTLE FOR EARTH HAS BEGIN... AFTER BRUNCH. This phrase humorously juxtaposes everyday activities (brunch) with epic events ("the battle"), hinting at a lighthearted tone despite its sci-fi setting.  Overall, combining these visual cues creates an intriguing blend between ordinary life and extraordinary circumstances, suggesting a comedic take on serious topics such as global conflict or invasion.
---
### Long list of tags
Describe with comma separated tags
> Independence Daze, Will Smith, action comedy, alien invasion, cityscape background, UFO, coffee cup, doughnut, urban destruction, patriotic theme, after brunch battle for Earth
---
### Short list of tags
In a comma separated list, caption with the top 3 tags
> Independence Daze, Will Smith, Sci-Fi Comedy
---
### Primary color
In one word: What is the primary color of this image.
> Blue
---
### Text (works sometimes)
`Write any large text in the image or answer with an empty string`
> INDEPENDENCE DAZE





## Credits
Thanks [Gökay Aydoğan](https://huggingface.co/gokaygokay) for helping me with the scripts.
