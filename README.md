### WHM CHALLENGE

[http://wmh.isi.uu.nl/](http://wmh.isi.uu.nl/)

The purpose of this challenge is to directly compare methods for the automatic segmentation of White Matter Hyperintensities (WMH) of presumed vascular origin.

DATASET CAN BE DOWNLOAD FROM [HERE](http://wmh.isi.uu.nl/data/)

### IMPROVEMENT

INPUT: 2 CHANNELS 240X240
OUTPUT: SEGMENTATION WITH LABEL 1 IN GROUND TROUTH

Unet with same padding instead of valid padding:
![](img/Unet.png)

1 replacement from "down-conv-up" to a atrous conv

![](img/Unet_with_dialate.png)
