module.exports = class Dense
{
  constructor(inSize, outSize, useBias, activation)
  {
    this.activation = activation;
    this.weights = tf.variable(tf.randomUniform([inSize, outSize], -1, 1), true);
    this.bias = useBias ? tf.variable(tf.zeros([outSize]), true) : null;
  }

  f(x)
  {
    x = x.rank == 1 ? x.expandDims() : x;

    x = x.matMul(this.weights);
    x = this.bias ? x.add(this.bias) : x;
    x = this.activation ? this.activation(x) : x;

    return x;
  }
}
