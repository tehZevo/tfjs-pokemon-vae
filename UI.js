module.exports = class UI
{
  constructor(app)
  {
    this.app = app;

    //textarea
    this.latentArea = document.getElementById("latent");
    this.canvas = document.getElementById("canvas");
    this.map = document.getElementById("map");
    this.g = canvas.getContext("2d");

    this.initUI();
    this.createSliders();
    //TODO: show the mons on dimension extremities :)
  }

  updateLatentUI()
  {
    //update text area
    this.latentArea.value = JSON.stringify(this.app.latent);

    //update sliders
    this.sliders.forEach((e, i) =>
    {
      e.slider.value = Math.tanh(this.app.latent[i] / this.app.sliderScale); //squash
    });
  }

  initUI()
  {
    $("#lr").innerHTML = "LR: " + this.app.lr;

    this.initListeners();
  }

  initListeners()
  {
    $("#verify").onclick = this.app.test.bind(this.app);
    $("#generate").onclick = this.app.generate.bind(this.app);
    //toggle training on/off
    $("#pause").onclick = () =>
    {
      this.app.doTrain = !this.app.doTrain;
      $("#pause").innerHTML = this.app.doTrain ? "Pause" : "Unpause";
    };

    $("#lrUp").onclick = () =>
    {
      this.app.changeLr(true);
      $("#lr").innerHTML = "LR: " + this.app.lr;
    }

    $("#lrDown").onclick = () =>
    {
      this.app.changeLr(false);
      $("#lr").innerHTML = "LR: " + this.app.lr;
    }

    this.map.onmousemove = (e) =>
    {
      var x = e.offsetX / this.map.width * 2 - 1;
      x = Math.atanh(x) * this.app.mapScale;
      var y = (1 - e.offsetY / this.map.height) * 2 - 1;
      y = Math.atanh(y) * this.app.mapScale;

      tf.tidy(() =>
      {
        this.app.latent[this.app.mapDimX] = x;
        this.app.latent[this.app.mapDimY] = y;
        this.updateLatentUI();
        this.drawLatent(this.app.latent);
      });

      //this.updateSliders();

      e.preventDefault();
    };
  }

  createSliders()
  {
    this.sliders = [];

    var sliderDiv = $("#sliders");
    for(var i = 0; i < this.app.latentSize; i++)
    {
      var min = document.createElement("canvas");
      var max = document.createElement("canvas");
      var closest = document.createElement("canvas");
      min.width = max.width = closest.width = this.app.imageWidth;
      min.height = max.height = closest.height = this.app.imageHeight;
      var slider = document.createElement("input");
      slider.type = "range";
      slider.min = -1;
      slider.max = 1;
      slider.step = "any";
      slider.style.width = "80%"
      slider.oninput = this.createSliderListener(i, slider);
      //append and add to slider array
      sliderDiv.appendChild(closest);
      sliderDiv.appendChild(min);
      sliderDiv.appendChild(slider);
      sliderDiv.appendChild(max);
      sliderDiv.appendChild(document.createElement("br"));

      var o = {};
      o.slider = slider;
      o.min = min;
      o.max = max;
      o.closest = closest;
      this.sliders.push(o);
    }

    return sliders;
  }

  createSliderListener(index, slider)
  {
    return () =>
    {
      this.app.latent[index] = Math.atanh(slider.value) * this.app.sliderScale;
      this.updateLatentUI();
      this.drawLatent(this.app.latent);
      this.updateSliders();
    };
  }

  drawLatent(latent)
  {
    tf.tidy(() =>
    {
      //TODO: move this to app since it involves vae?
      //decode latent
      var output = this.app.vae.decode(tf.tensor(latent));
      //draw image
      U.drawImage(output, this.app.imageWidth, this.app.imageHeight, this.canvas);
    });
  }

  updateSliders(which)
  {
    //TODO
    var mins = [];
    var minVals = [];
    var maxs = [];
    var maxVals = [];
    var closests = [];
    var closestVals = [];

    //for each image
    Object.keys(this.app.latents).forEach((index) =>
    {
      var latent = this.app.latents[index];
      //console.log(which)
      var start = which == null ? 0 : which;
      var stop = which == null ? this.app.latentSize : which + 1;
      //for each latent dimension
      for(var i = start; i < stop; i++)
      {
        var latentValue = latent[i]; //value of this latent dimension in the image
        var minVal = minVals[i]; //minimum value of this latent dimension so far
        var maxVal = maxVals[i];
        var closestVal = closestVals[i];
        var distToSlider = Math.abs(latentValue - this.sliders[i].slider.value);

        if(minVal == null || latentValue < minVal)
        {
          minVals[i] = latentValue;
          mins[i] = index;
        }
        if(maxVal == null || latentValue > maxVal)
        {
          maxVals[i] = latentValue;
          maxs[i] = index;
        }

        if(closestVal == null || distToSlider < closestVal)
        {
          closestVals[i] = distToSlider;
          closests[i] = index;
        }
      }

      //loop over each latent dimension and draw images to canvases
      for(var i = 0; i < this.app.latentSize; i++)
      {
        var o = this.sliders[i];
        var minCanvas = o.min;
        var maxCanvas = o.max;
        var closestCanvas = o.closest;
        //console.log(mins[i], maxs[i], closests[i])
        var gMin = minCanvas.getContext("2d");
        gMin.clearRect(0, 0, minCanvas.width, minCanvas.height);
        gMin.drawImage(this.app.images[mins[i]].image, 0, 0);
        var gMax = maxCanvas.getContext("2d");
        gMax.clearRect(0, 0, maxCanvas.width, maxCanvas.height);
        gMax.drawImage(this.app.images[maxs[i]].image, 0, 0);
        var gClose = closestCanvas.getContext("2d");
        gClose.clearRect(0, 0, closestCanvas.width, closestCanvas.height);
        gClose.drawImage(this.app.images[closests[i]].image, 0, 0);
      }

    });
  }

  updateMap()
  {
    var g = map.getContext("2d");
    g.fillStyle = "rgb(255, 255, 255)";
    g.fillRect(0, 0, map.width, map.height);

    Object.keys(this.app.latents).forEach((i) =>
    {
      var img = this.app.images[i].image;
      var latent = this.app.latents[i];
      //TODO: do this when inserting into latents object??
      latent = latent.map((e) => Math.tanh(e / this.app.mapScale));

      //0, 0 == bottom left
      var x = (latent[0] / 2 + 0.5) * this.map.width;
      var y = this.map.height - (latent[1] / 2 + 0.5) * this.map.height;

      g.drawImage(img, x - this.app.imageWidth / 2, y - this.app.imageHeight / 2);
    });

  }
}

$ = document.querySelector.bind(document);
