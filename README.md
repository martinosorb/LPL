# LPL

This code is an unofficial implementation of the paper

```
Halvagal, M. S., & Zenke, F. (2022). The combination of Hebbian and predictive plasticity learns invariant object representations in deep sensory networks. bioRxiv.
```

currently, the predictive loss does not seem to work. I managed to qualitatively reproduce the results of the paper
on CIFAR10 and STL10 *without* predictive loss, only relying on the Hebbian and decorrelation terms.

## Usage
Install requirements: `pip install -r requirements.txt`.

You can train a network using `train_vgg.py` (parameters are currently to be changed within the file).
A pretrained network (without using the predictive loss) can be found in the models folder.

You can then train a decoder from each layer of the trained network with
```python linear_train.py --name models/<model>.pth --avgpool```
where `avgpool` applies global average pooling before the decoder.

Then, the result can be plotted with
```python make_figure.py reports/*```
which will create a PNG image.
