# A tour of the TUI

`utrain tui` is the whole of utrain in one screen: the runs you have, the phases
each is made of, the config it was started with, its curves as they fill, and
its log. This is a recording of one run from nothing to a conversation with the
model it produced.

<div id="tour" data-cast="../assets/tour.cast" data-poster="npt:0:45"></div>

## What goes past

**`i` and `c`** are the other two things utrain knows about: the images you have
pulled, and the CPU and GPUs it can put a run on. Both are a keypress from
anywhere, and `escape` comes back.

**`n`** creates a run. A run needs three things — a name, an image and something
to run on — and everything else about it is config.

**`2`** is the config utrain wrote from the image's schema. `down` moves the
cursor, `e` edits the field it is on, and `enter` writes it straight to
`runs/<id>/config.yaml`. Here the model is taken from nanochat's default depth
of 8 down to 6, which is the first thing to do to a run that has one GPU rather
than eight.

**`s`** starts it, and `enter` on the run opens the phases it is made of —
download, tokenizer, pretrain, sft, rl — which go from pending to running to
done in order.

**`3`** is the plots. Each phase declares which curves are worth drawing, and
they follow the run live. `m` opens the metric picker: `space` draws a metric or
stops drawing it, `y` solos one, `l` switches to a log y axis and `b` swaps the
half-block glyphs for braille, which is finer where the font has it. `E` writes
the curve out as csv, png, svg or pdf.

**`4`** is the phase's log, tailed as it is written.

**`t`** talks to what the run produced. The phase under the cursor is the one it
serves and the reply streams back from an OpenAI-compatible
endpoint the container brings up for as long as the screen is open. `escape`
ends the session and stops the container.

**`?`** lists every key, and `q` quits.

## Doing it yourself

The [quickstart](quickstart.md) walks through the same arc from the command
line, against a real image that trains a real model in a couple of minutes. The
TUI is another way into exactly the same runs: anything started here can be
inspected with `utrain run show`, and anything started there shows up in the
list above.
