module.exports = class VAE
{
  constructor(imageWidth, imageHeight, imageChannels, latentSize, dropout)
  {
    //TODO: channel multiplier? (default to 2)
    //TODO: readd dropout
    this.imageWidth = imageWidth;
    this.imageHeight = imageHeight;
    this.imageChannels = imageChannels;
    this.latentSize = latentSize;
    this.dropout = dropout;

    var convActi = tf.relu; //this.softsign
    var bilinear = true;
    var enc = [];
    enc.push(new Conv2D(3, 3, 10, 16, 16, convActi, bilinear));
    enc.push(new Conv2D(3, 10, 20, 8, 8, convActi, bilinear));
    enc.push(new Conv2D(3, 20, 30, 4, 4, convActi, bilinear));
    enc.push(new Conv2D(3, 30, 40, 2, 2, convActi, bilinear));
    enc.push(new Conv2D(3, 40, 50, 1, 1, convActi, bilinear));
    this.encodeLayers = enc;

    //TODO: latent layers (post-conv size to latent size and back)
    //192 -> latentSize
    var acti = null;
    this.encMean = new Dense(50, this.latentSize, true, acti);
    this.encStd = new Dense(50, this.latentSize, true, acti);

    this.decSample = new Dense(this.latentSize, 50, true, this.softsign); //TODO: relu or what..?

    var dec = [];
    dec.push(new Conv2D(3, 50, 50, 1, 1, convActi, bilinear));
    dec.push(new Conv2D(3, 50, 40, 2, 2, convActi, bilinear));
    dec.push(new Conv2D(3, 40, 30, 4, 4, convActi, bilinear));
    dec.push(new Conv2D(3, 30, 20, 8, 8, convActi, bilinear));
    dec.push(new Conv2D(3, 20, 10, 16, 16, convActi, bilinear));
    dec.push(new Conv2D(3, 10, 3, imageWidth, imageHeight, this.softSign, bilinear));
    this.decodeLayers = dec;

    //heavily penelize from straying from unit gaussian
    this.klScale = tf.scalar(10);

    this.optimizer = tf.train.adam(0.001);
  }

  softsign(x)
  {
    return x.div(tf.ones(x.shape).add(x.abs()));
  }

  auto(x)
  {
    var [mean, logStd] = this.encode(x);
    var latent = this.sample(mean, logStd);
    var pred = this.decode(latent);

    return pred;
  }

  //generates mean and log(std) vectors
  encode(x)
  {
    this.encodeLayers.forEach((e) => x = e.f(x));
    //console.log(x.shape)
    //console.log(x.dataSync()[0]);
    x = x.squeeze(); //squeeze extra dims out of x left over from width/height
    var mean = this.encMean.f(x);
    var logStd = this.encStd.f(x);

    return [mean, logStd];
  }

  //creates a sample of the means and stds provided
  sample(mean, logStd)
  {
    //console.log(mean.dataSync()[0], logStd.dataSync()[0]);
    //we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
    var norm = tf.randomNormal(mean.shape);
    //sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    var sample = mean.add(logStd.exp().mul(norm));
    //console.log("mean: " + mean.mean().dataSync()[0], "logStd: " + logStd.mean().dataSync()[0], "sample: " + sample.mean().dataSync()[0]);
    return sample;
  }

  decode(x)
  {
    x = x.rank == 1 ? x.expandDims() : x;

    //latentSize -> 192 or whatever
    x = this.decSample.f(x);

    //add width/height
    x = x.expandDims(1);
    x = x.expandDims(1);

    //TODO: deconv here
    this.decodeLayers.forEach((e) => x = e.f(x));

    //x = x.clipByValue(-1, 1); //allow relus to go crazy
    return x;
  }

  fLoss(pred, target, mean, logStd)
  {
    //compute the average MSE error, then scale it up, ie. simply sum on all axes
    var reconstructionLoss = tf.sum(tf.square(pred.sub(target)));
    //compute the KL loss
    //kl_loss = - 0.5 * K.sum(1 + log_stddev - K.square(mean) - K.square(K.exp(log_stddev)), axis=-1)
    var meanSquared = mean.square();
    var stdSquared = logStd.exp().square();
    var onePlusLogStd = tf.scalar(1).add(logStd);
    var sum = tf.sum(onePlusLogStd.sub(meanSquared).sub(stdSquared), 1);
    var klLoss = tf.scalar(-0.5).mul(sum);
    //TODO: scale kl loss to further penalize straying from unit gaussian?
    klLoss = klLoss.mul(this.klScale);
    //return the average loss over all images in batch
    var totalLoss = tf.mean(reconstructionLoss.add(klLoss));
    return totalLoss;
  }

  train(batch, returnLatents)
  {
    //stack em up
    var batchTensor;
    if(batch.length == 1)
    {
      batchTensor = batch[0];
    }
    else
    {
      batchTensor = tf.stack(batch);
    }

    var latents = [];
    var loss = this.optimizer.minimize(() =>
    {
      //get mean/logstd vectors
      var [mean, logStd] = this.encode(batchTensor);

      //grab latent samples Norm(mean, logstd)
      var latent = this.sample(mean, logStd);

      if(returnLatents)
      {
        //i think 0 is the right axis
        latents = tf.split(mean, batch.length, 0).map((e) => Array.from(e.dataSync()));
      }

      //decode samples
      var pred = this.decode(latent);

      return this.fLoss(pred, batchTensor, mean, logStd);
    }, true).dataSync()[0];

    return [latents, loss];
  }
}

var Conv2D = require("./Conv2D.js");
var Dense = require("./Dense.js");
