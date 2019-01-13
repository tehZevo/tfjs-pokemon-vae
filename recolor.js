var fs = require("fs");
var Jimp = require("jimp");

//var inPath = "./unown-orig"
var inPath = "./game-icons.net.png"
var outPath = "./gameicons-32";

var outW = 32;
var outH = 32;
var bw = false;
var bg = 0x808080ff;

fs.readdir(inPath, (err, files) =>
{
  files.forEach((file) =>
  {
    if(file.endsWith(".png"))
    {
      Jimp.read(inPath + "/" + file, (err, image) =>
      {
        if(bw) { image.grayscale(); }
        image.scaleToFit(outW, outH); //scale

        new Jimp(outW, outH, bg, (err, bg) =>
        {
          //draw pokemon onto white background
          bg.composite(image, (outW - image.bitmap.width) / 2, (outH - image.bitmap.height) / 2);
          //bg.composite(image, 0, 0);
          if(bw) { bg.grayscale(); }

          bg.write(outPath + "/" + file);
          console.log("done " + file);
        });


      });
    }
  });
});
