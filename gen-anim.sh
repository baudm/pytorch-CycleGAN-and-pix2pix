#!/bin/sh

convert -delay 10 -loop 0 `seq 20000 100000 6260000 | sed 's|$|.png|'` -crop 512x512+0+0 +repage out.gif
