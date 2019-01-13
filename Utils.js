var U = {};

module.exports = U;

/** draws a tensor formatted as above to the canvas */
U.drawImage = function(imageTensor, width, height, canvas)
{
  var g = canvas.getContext("2d");
  var it = imageTensor;

  tf.tidy(() =>
  {
    //TODO: dimension direction.. y first? idk
    it = it.add(tf.ones(it.shape));
    it = it.div(tf.scalar(2));
    it = it.mul(tf.scalar(255));
    //downsample
    it = it.resizeBilinear([height, width]);
    it = it.dataSync();
    //var maxX = 0;
    //var maxY = 0;
    for(var i = 0; i < it.length; i++)
    {
      var x = i % (width);
      var y = Math.floor(i / (width));
      var cr = Math.round(it[i * 3]);
      var cg = Math.round(it[i * 3 + 1]);
      var cb = Math.round(it[i * 3 + 2]);
      //maxX = Math.max(maxX, x);
      //maxY = Math.max(maxY, y);
      g.fillStyle = "rgb(" + cr + "," + cg + "," + cb + ")";
      g.fillRect(x, y, 1, 1);
    }
  });

  //console.log(maxX, maxY, i)
}

U.getBatch = function(images, size)
{
  var ids = [];
  var batch = [];
  for(var i = 0; i < size; i++)
  {
    var index = Math.floor(Math.random() * images.length);
    batch.push(images[index].tensor);
    ids.push(index);
  }

  if(batch.length == 0)
  {
    return null;
  }

  return [ids, batch];
}
