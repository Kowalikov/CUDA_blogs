# Prerequisities:

All in the main `./setup.md` + [Imagemagick](https://imagemagick.org/).
```
sudo apt install imagemagick
```

# Running the demo:


```
nvcc raytrace.cu -o raytrace
./raytrace
convert raytraced_frame.ppm raytraced_frame.png
```