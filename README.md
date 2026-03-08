# Screen Damage Detection

This project used to to assist in buying and selling refurbished phones by analyzing the phone’s screen condition from an image.

First, a YOLO model is used to detect the phone as where it located in the image.(A special thanks to the 
DATACLUSTER LABS Kaggle, whom dataset is used to train the model.) 
Then, a pre-trained Mask R-CNN model performs precise segmentation of the detected phone(remove the backgrounds). 
Finally, a Canny edge detector is applied to estimate the percentage of screen damage based on the detected cracks and scratches areas.

I provided a video for seeing the performance and use to with the method as how we can use this.

along with that one can run the streamlit.py in their system to use this with GUI.

and seg_img_damage_calcultion.py run it in the terminal
