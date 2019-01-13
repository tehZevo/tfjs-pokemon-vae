It's a mess. :^)

You'll need Electron, eg: `npm install -g electron`
You'll also need some pokemon sprites, they can be sourced from veekun database: https://veekun.com/dex/downloads under "Others" -> "Pokemon icons"

The icons you download may have unexpected colors in transparent regions (ie where a = 0, r/g/b != 0), so that may effect VAE outcome.

To run:
* Populate a folder with 40x30 pokemon icons
* Edit App.js "this.path" line to reference that folder
* `npm start` to run
