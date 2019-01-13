var fs = require("fs");
var tf = require("@tensorflow/tfjs-core")
var VAE = require("./VAE.js");

//TODO: use image <-> tensor functions to speed up drawing
class App
{
  constructor()
  {
    this.images = [];
    this.batchSize = 100;
    this.imageWidth = 40;
    this.imageHeight = 30;
    this.superSample = 1;
    this.imageChannels = 3;
    this.latentSize = 2;
    this.sliderScale = 5; //TODO: auto slider scale?
    this.mapUpdateTicks = 1; //ever N ticks, draw all mons on map
    this.sliderUpdateTicks = 10; //every N ticks, update slider images
    this.mapScale = 7;
    this.mapDimX = 0;
    this.mapDimY = 1;
    //location of images (pokemon/unown)
    //this.path = "./unown-alpha-gray";
    //this.path = "./pokemon-151-alpha-gray";
    this.path = "./pokemon-alpha-gray";
    //TODO: reimplement dropout?
    this.dropout = 0.0;

    this.lossLerp = null;
    this.dtLerp = null;
    this.lerpRate = 0.01;
    this.doTrain = true;
    this.ticks = 0;

    //latent vectors for each image
    this.latents = {};

    //active latent vector
    this.latent = Array.apply(null, Array(this.latentSize)).map(Number.prototype.valueOf, 0);

    this.init();

    setTimeout(this.tick.bind(this), 1000);
  }

  init()
  {
    //start loading files
    fs.readdir(this.path, (err, files) =>
    {
      files.forEach((file) =>
      {
        if(file.endsWith(".png"))
        {
          var img = new Image();
          img.src = this.path + "/" + file;
          img.onload = () => this.load(img);
        }

        //console.log(file);
      });
    });

    //create VAE
    this.vae = this.createVAE();

    this.lr = 0.0001;
    this.vae.optimizer = tf.train.adam(this.lr);
    //penelize from straying from unit gaussian
    this.vae.klScale = tf.scalar(0.1);

    //create UI manager
    this.ui = new UI(this);
  }

  createVAE()
  {
    return new VAE(
      this.imageWidth * this.superSample,
      this.imageHeight * this.superSample,
      this.imageChannels,
      this.latentSize,
      this.dropout);
  }

  //generate from a random latent vector
  generate()
  {
    tf.tidy(() =>
    {
      this.latent = Array.from(tf.randomNormal([this.latentSize]).dataSync());
      this.ui.updateLatentUI();
      this.ui.drawLatent(this.latent);
    });
  }

  changeLr(upDown)
  {
    this.lr = upDown ? this.lr * 1.1 : this.lr / 1.1;
    this.vae.optimizer = tf.train.adam(this.lr);
  }

  //test the vae with an actual mon
  test()
  {
    tf.tidy(() =>
    {
      var [_, image] = U.getBatch(this.images, 1);
      image = image[0]; //first image in batch
      var [mean, logStd] = this.vae.encode(image);
      var localLatent = this.vae.sample(mean, logStd);
      //show pokemon's latent value
      this.latent = Array.from(localLatent.dataSync());

      var output = this.vae.decode(localLatent);

      this.ui.updateLatentUI();
      U.drawImage(output, this.imageWidth, this.imageHeight, this.ui.canvas);
    });
  }

  async tick()
  {
    var t = +new Date();

    if(this.doTrain)
    {
      tf.tidy(() =>
      {
        //grab batch
        var [ids, batch] = U.getBatch(this.images, this.batchSize);

        if(batch == null)
        {
          console.log("no data")
        }
        else
        {
          //grab latents from training process (so we dont have run the net twice)
          var [newLatents, loss] = this.vae.train(batch, true);

          //update latents
          //for each id
          ids.forEach((e, i) =>
          {
            //replace old latent with new one
            //TODO: store latents in images structure?
            this.latents[e] = newLatents[i];
          });



          this.lossLerp = this.lossLerp == null ? loss : this.lossLerp;
          this.lossLerp += (loss - this.lossLerp) * this.lerpRate;


          console.log("iteration " + (this.ticks + 1) + " loss (smooth): " + this.lossLerp.toExponential());
        }

        //dispose batch
        //batch.dispose();
      });

      if(this.ticks % this.mapUpdateTicks == 0)
      {
        this.ui.updateMap();
      }

      if(this.ticks % this.sliderUpdateTicks == 0)
      {
        //this.ui.updateSliders();
      }
    }

    var dt = (+new Date()) - t;
    dt /= 1000;
    this.dtLerp = this.dtLerp == null ? dt : this.dtLerp;
    this.dtLerp += (dt - this.dtLerp) * this.lerpRate;

    console.log("mem: " + tf.memory().numBytes +
      ", tensors: " + tf.memory().numTensors +
      ", " + this.dtLerp.toFixed(2) + "s");

    this.ticks++;

    setTimeout(this.tick.bind(this), 0);
  }

  load(img)
  {
    var tensor = tf.fromPixels(img, this.imageChannels);
    tensor = this.formatImage(tensor);

    console.log("loaded " + img.src);

    //TODO: add latents here
    this.images.push({image: img, tensor: tensor});
  }

  /** pushes pixels to -1 to 1 range */
  formatImage(tensor)
  {
    tensor = tensor.asType("float32");
    tensor = tensor.div(tf.scalar(255));
    tensor = tensor.mul(tf.scalar(2));
    tensor = tensor.sub(tf.ones(tensor.shape));
    tensor = tensor.resizeNearestNeighbor([
      this.imageHeight * this.superSample,
      this.imageWidth * this.superSample]);

    return tensor;
  }
}

//TODO: vae operations (add, subtract, mix)
//TODO: save/load latent features
//TODO: subtracting mons to generate feature vectors
//TODO: adding mon + feature vectors

var UI = require("./UI.js");
var U = require("./Utils.js");

new App();
