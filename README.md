
INPUT: 2 CHANNELS 240X240
OUTPUT: SEGMENTATION WITH LABEL 1 IN GROUND TROUTH

Unet with same padding instead of valid padding:
![](img/Unet.png)

1 replacement from "down-conv-up" to a atrous conv

![](img/Unet_with_dialate.png)
