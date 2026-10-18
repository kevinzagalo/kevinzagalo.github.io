## Generate and publish the website

The current design is generated from `index.jemdoc`, `publications.bib`, `MENU`,
and the images in `img/` with:

```sh
python3 new_jemdoc.py -o new_index.html index.jemdoc
```

Every push to `main` runs `new_jemdoc.py` in GitHub Actions and publishes the
result to GitHub Pages. Generated HTML is not tracked in the repository.

The original `jemdoc.py` and `jemdoc.css` are kept unchanged. To preview the
previous renderer, run:

```sh
python3 jemdoc.py -o legacy_index.html index.jemdoc
```

## Original jemdoc project

jemdoc is a light text-based markup language designed for creating websites. It
takes a text file written with jemdoc markup, an optional configuration file and
an optional menu file, and makes static websites that look something like
http://jemdoc.jaboc.net/.

It was written by me, Jacob Mattingley, in 2007, and definitely isn't the type
of code I would put on my CV. Lots of people use jemdoc, especially in academia.

Much more info at http://jemdoc.jaboc.net/.
