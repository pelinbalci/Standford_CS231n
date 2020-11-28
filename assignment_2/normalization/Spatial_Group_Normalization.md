Ref: https://colab.research.google.com/drive/16_iX-A0lmuvoOL1zkdzmJjmLJ0hmRyET#scrollTo=_rDdIUtvcu7C

# Spatial Normalization 

*Note: Copied from colab*

We already saw that batch normalization is a very useful technique for training deep fully-connected networks. 
As proposed in the original paper (link in BatchNormalization.ipynb), batch normalization can also be used for 
convolutional networks, but we need to tweak it a bit; the modification will be called "spatial batch normalization."

Normally batch-normalization accepts inputs of shape (N, D) and produces outputs of shape (N, D), where we normalize 
across the minibatch dimension N. For data coming from convolutional layers, batch normalization needs to accept 
inputs of shape (N, C, H, W) and produce outputs of shape (N, C, H, W) where the N dimension gives the minibatch size 
and the (H, W) dimensions give the spatial size of the feature map.

If the feature map was produced using convolutions, then we expect every feature channel's statistics e.g. mean, 
variance to be relatively consistent both between different images, and different locations within the same image -- 
after all, every feature channel is produced by the same convolutional filter! Therefore spatial batch normalization 
computes a mean and variance for each of the C feature channels by computing statistics over the minibatch dimension N 
as well the spatial dimensions H and W.

[1] Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
 Covariate Shift", ICML 2015.


# Group Normalization

*Note: Copied from colab*

In the previous notebook, we mentioned that Layer Normalization is an alternative normalization technique that mitigates
 the batch size limitations of Batch Normalization. However, as the authors of [2] observed, Layer Normalization 
 does not perform as well as Batch Normalization when used with Convolutional Layers:

With fully connected layers, all the hidden units in a layer tend to make similar contributions to the final prediction, 
and re-centering and rescaling the summed inputs to a layer works well. However, the assumption of similar contributions 
is no longer true for convolutional neural networks. The large number of the hidden units whose receptive fields lie 
near the boundary of the image are rarely turned on and thus have very different statistics from the rest of the hidden units 
within the same layer.

The authors of [3] propose an intermediary technique. In contrast to Layer Normalization, where you normalize 
over the entire feature per-datapoint, they suggest a consistent splitting of each per-datapoint feature into G groups, 
and a per-group per-datapoint normalization instead.

Even though an assumption of equal contribution is still being made within each group, the authors hypothesize that 
this is not as problematic, as innate grouping arises within features for visual recognition. One example they use to 
illustrate this is that many high-performance handcrafted features in traditional Computer Vision have terms that are 
explicitly grouped together. Take for example Histogram of Oriented Gradients [4]-- after computing histograms per 
spatially local block, each per-block histogram is normalized before being concatenated together to form the final 
feature vector.

You will now implement Group Normalization. Note that this normalization technique that you are to implement in the 
following cells was introduced and published to ECCV just in 2018 -- this truly is still an ongoing and excitingly 
active field of research!

[2] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." stat 1050 (2016): 21.
[3] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In Computer Vision and Pattern Recognition (CVPR), 2005.


## CNN Application

Ref: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html#torch.nn.GroupNorm [torch]
Ref: https://www.tensorflow.org/addons/tutorials/layers_normalizations [tf]

torch.nn.GroupNorm(num_groups: int, num_channels: int, eps: float = 1e-05, affine: bool = True)

- The input channels are separated into num_groups groups, each containing num_channels / num_groups channels.
- The mean and standard-deviation are calculated separately over the each group.
- The standard-deviation is calculated via the biased estimator, equivalent to torch.var(input, unbiased=False).

        - num_groups (int) – number of groups to separate the channels into
        - num_channels (int) – number of channels expected in input [torch
        
GN experimentally scored closed to batch normalization in image classification tasks. 
It can be beneficial to use GN instead of Batch Normalization in case your overall batch_size is low, 
which would lead to bad performance of batch normalization. [tf]


## Group Normalizaiton Paper Notes

[3] Wu, Yuxin, and Kaiming He. "Group Normalization." arXiv preprint arXiv:1803.08494 (2018).

BN’s error increases rapidly when the batch size becomes smaller, caused by inaccurate batch
statistics estimation.

GN divides the channels into groups and computes within each group the mean and variance for normalization. GN’s computation 
is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes.

## Related Works:*

Layer Normalization (LN) operates along the channel dimension, and Instance Normalization (IN) [61] performs BN-like computation but only for each sample (Figure 2). Instead of operating on features, Weight Normalization (WN) [51] proposes to normalize the filter weights.
These methods do not suffer from the issues caused by the
batch dimension, but they have not been able to approach
BN’s accuracy in many visual recognition tasks.

#### BN:

    Si is the set of pixels in which the mean and std are computed, and m is the size of this set.
    
    Si_BN = {k | k_c = i_c
    
    where iC (and kC ) denotes the sub-index of i (and k) along the C axis. 

This means that the pixels sharing the same channel index are normalized together, i.e., 
for each channel, BN computes µ and σ along the (N, H, W) axes. 


#### LN:

    Si = {k | kN = iN }
 
LN computes µ and σ along the (C, H, W) axes for each sample.

## GN: 

- G is the number of groups :  it is a pre-defined hyper-parameter (G = 32 by default).
- C/G is the number of channels per group. 
- [k_C / C/G] = [i_C / C/G] means that the indexes i and k are in
the same group of channels, assuming each group of channels are stored in a sequential order along the C axis
- GN computes µ and σ along the (H, W) axes and along a group of C/G channels. 


    Si = {k | k_N = i_N, [k_C / C/G] = [i_C / C/G] }
