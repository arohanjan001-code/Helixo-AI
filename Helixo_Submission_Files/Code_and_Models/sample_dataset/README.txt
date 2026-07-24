The full dataset is too large to upload here (6000+ raw images, 12,198 after augmentation).

What's in the dataset:
- Full face helmets, half face helmets (visor up and visor down)
- Bare heads, caps, turbans, bandanas (these were the hard cases)
- Photos in daylight, dusk, and night with IR lighting
- Images taken from the actual mounting angle on bike instrument cluster

Dataset was built and annotated using Roboflow.
Augmentation included flips, rotation, crop, brightness/contrast jitter, blur, noise, and mixup.

The trained model is in /AI_Models/
