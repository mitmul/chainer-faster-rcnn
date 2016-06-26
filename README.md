# Faster R-CNN

This is an experimental implementation of Faster R-CNN using Chainer based on Ross Girshick's [py-faster-rcnn codes](https://github.com/rbgirshick/py-faster-rcnn).

## Requirement

Using anaconda is strongly recommended.

- Python 2.7.6+, 3.4.3+, 3.5.1+

  - [Chainer](https://github.com/pfnet/chainer) 1.9.1+
  - NumPy 1.9, 1.10, 1.11
  - Cython 0.23+
  - OpenCV 2.9+, 3.1+

### Installation of dependencies

```
pip install numpy
pip install cython
pip install chainer
# for python3
conda install -c https://conda.binstar.org/menpo opencv3
# for python2
conda install opencv
```

### For Windows users

There's a known problem in cpu_nms.pyx. But a workaround has been posted [here](https://github.com/mitmul/chainer-faster-rcnn/issues/1) (and see also [the issue posted to the original py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn/issues/36)).

## Inference

### 1\. Download pre-trained model

```
wget https://www.dropbox.com/s/2fadbs9q50igar8/VGG16_faster_rcnn_final.model?dl=0
mv VGG16_faster_rcnn_final.model?dl=0 VGG16_faster_rcnn_final.model
```

### 2\. Build extensions

```
cd lib
python setup.py build_ext -i
```

### 3\. Use forward.py

```
wget http://vision.cs.utexas.edu/voc/VOC2007_test/JPEGImages/004545.jpg

python forward.py --img_fn 004545.jpg --gpu 0
```

`--gpu 0` turns on GPU. When you turn off GPU, use `--gpu -1` or remove `--gpu` option.

![](https://raw.githubusercontent.com/wiki/mitmul/chainer-faster-rcnn/images/result.png)

## Training

will be updated soon

## Framework

![](https://raw.githubusercontent.com/wiki/mitmul/chainer-faster-rcnn/images/Faster%20R-CNN.png)
