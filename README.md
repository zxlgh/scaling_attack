## Camouflage Tutorial

process of generation camouflage image is following:

```python
####  pseudocode

# load images
src_img, tar_imgs = load_image()
# declare the scaler as many as target images
scaler = PillowScaler(algorithm, src_shape, tar_shape)
...
# declare the attacker
attacker = Attack(src_file, [tar_files], [scalers])
# att is the camouflage image
att = attacker.attack()
```
