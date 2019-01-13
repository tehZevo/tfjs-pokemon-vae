//convolves and resizes
module.exports = class Conv2D
{
  constructor(filterSize, inChannels, outChannels, outWidth, outHeight, activation, bilinear)
  {
    this.activation = activation;
    bilinear = bilinear || false;

    this.bilinear = bilinear;

    this.outWidth = outWidth;
    this.outHeight = outHeight;

    var filterShape = [filterSize, filterSize, inChannels, outChannels];

    //if relus give nan, try decreasing the range here
    this.filter = tf.variable(tf.randomUniform(filterShape, -.1, .1), true);
    //this.filter = tf.variable(tf.randomNormal(filterSize), true);
    this.bias = tf.variable(tf.zeros([1]), true);
  }

  f(x)
  {
    //var before = x.shape.slice();

    x = x.conv2d(this.filter, 1, "same");
    x = x.add(this.bias); //broadcast!
    //console.log(x.mean().dataSync()[0]);
    x = this.activation ? this.activation(x) : x;

    //resize
    var op = this.bilinear ? tf.image.resizeBilinear : tf.image.resizeNearestNeighbor;
    x = op(x, [this.outHeight, this.outWidth]);

    //console.log("downconv [" + before.join(",") + "] -> [" + x.shape.join(",") + "]");
    //console.log(x.mean().dataSync()[0]);

    return x;
  }
}
