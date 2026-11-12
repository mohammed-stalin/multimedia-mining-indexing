# Colour spaces

`example.py` splits an image into its channels in three colour spaces and shows
them in a grid:

| Space | Channels | Typical use |
|---|---|---|
| BGR | Blue, Green, Red | OpenCV's native order (matplotlib expects RGB!) |
| HSV | Hue (0-179 in OpenCV), Saturation, Value | colour thresholding independent of brightness |
| L\*a\*b\* | Lightness, green-red, blue-yellow | perceptually uniform colour distances |

## Run

```bash
python example.py               # uses ../assets/rose.jpg
python example.py my_image.jpg
```
