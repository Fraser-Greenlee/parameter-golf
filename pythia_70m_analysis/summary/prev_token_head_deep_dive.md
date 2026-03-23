# L2_H1 Previous-Token Head: Deep Dive

Comparing trained vs engineered (bias-based + trained V/O) on 40 FineWeb docs.

## Attention Distribution Statistics

### Trained
- Total positions: 10482
- Top-1 is prev-token: 99.2%
- Prev-token weight: mean=0.7708, median=0.7832, min=0.0094, p5=0.5682, p95=0.9388
- Self weight: mean=0.0933, median=0.0819
- Entropy: mean=0.9199, median=0.9119, p5=0.3142, p95=1.5574

### Engineered
- Total positions: 10482
- Top-1 is prev-token: 100.0%
- Prev-token weight: mean=0.9968, median=0.9968, min=0.9968, p5=0.9968, p95=0.9968
- Self weight: mean=0.0016, median=0.0016
- Entropy: mean=0.0237, median=0.0238, p5=0.0238, p95=0.0238

## Prev-Token Weight Distribution

| Bucket | Trained | Engineered |
|--------|---------|------------|
| 0.00-0.50 | 2.2% | 0.0% |
| 0.50-0.80 | 54.4% | 0.0% |
| 0.80-0.90 | 31.1% | 0.0% |
| 0.90-0.95 | 9.0% | 0.0% |
| 0.95-0.99 | 3.2% | 0.0% |
| 0.99-1.00 | 0.1% | 100.0% |

## Where Trained L2_H1 Does NOT Attend to Previous Token

Found 83 positions (0.8% of total)

### Top 30 Non-Prev-Token Positions (by entropy)

| Token | Prev Token | Context | Top-1 Attended | Prev Weight | Entropy |
|-------|-----------|---------|---------------|-------------|---------|
| `t` | `’` |  If  your  city  doesn ’ [t]  do  this | ` doesn`@275 (0.28) | 0.212 | 2.504 |
| `.` | ` humor` |  had  a  sense  of  humor [.]  Kn aus | `.`@276 (0.33) | 0.294 | 2.368 |
| `t` | `’` | ,  and  it  doesn ’ [t]  fill  the | ` doesn`@228 (0.25) | 0.241 | 2.349 |
| `t` | `’` |  and  if  it  didn ’ [t]  exist - | ` didn`@215 (0.35) | 0.287 | 2.169 |
| `,”` | ` happened` |  how  this  could  have  happened [,”]  he  told | `,”`@156 (0.34) | 0.303 | 2.157 |
| `,”` | `ery` |  del i  and  bak ery [,”]  said  Walsh | `,”`@204 (0.37) | 0.328 | 2.061 |
| `t` | `’` | .  “ I  don ’ [t]  understand  how | ` don`@147 (0.36) | 0.292 | 1.888 |
| `t` | `’` | .  Now  this  wouldn ’ [t]  have  been | ` wouldn`@64 (0.40) | 0.261 | 1.883 |
| ` as` | ` such` |  by  this  combo ,  such [ as]  a  single | `,`@185 (0.37) | 0.281 | 1.863 |
| `.` | ` purposes` |  for  educational  and  academic  purposes [.]  It  may | `.`@188 (0.38) | 0.364 | 1.826 |
| `t` | `’` |  also  claim  they  weren ’ [t]  the  ones | ` weren`@235 (0.40) | 0.349 | 1.768 |
| `s` | `’` |  Wisconsin .  Bow man ’ [s]  strong  performance | `man`@92 (0.32) | 0.291 | 1.758 |
| `s` | `’` |  was  giving  Johannes en ’ [s]  estate  a | `en`@79 (0.42) | 0.238 | 1.756 |
| `selling` | `-` |  were  inevitable .  Best - [selling]  economist  Robert | ` Best`@240 (0.45) | 0.268 | 1.667 |
| `s` | `’` | ,  tell  them  it ’ [s]  a  great | ` it`@284 (0.51) | 0.267 | 1.601 |
| `s` | `’` |  people  share ,  it ’ [s]  worth  the | ` it`@19 (0.42) | 0.303 | 1.557 |
| `s` | `’` |  “ While  addressing  Ohio ’ [s]  need  for | ` Ohio`@240 (0.50) | 0.283 | 1.547 |
| `s` | `’` | ?  I  know  it ’ [s]  not  nice | ` it`@97 (0.39) | 0.374 | 1.543 |
| `t` | `’` |  but  crypt ically  can ’ [t]  reveal  the | ` can`@250 (0.49) | 0.286 | 1.526 |
| `s` | `’` |  away .  But  that ’ [s]  not  what | ` that`@92 (0.57) | 0.148 | 1.523 |
| `s` | `’` |  of  the  past  day ’ [s]  on - | ` day`@88 (0.44) | 0.356 | 1.452 |
| `s` | `’` | Neil  reported :  Seattle ’ [s]  draft  class | ` Seattle`@89 (0.51) | 0.259 | 1.435 |
| `part` | `-` | - do - us - [part]  MAR RI | `us`@180 (0.44) | 0.410 | 1.408 |
| `t` | `’` |  world :  Why  can ’ [t]  people  share | ` can`@136 (0.42) | 0.342 | 1.402 |
| `s` | `’` |  that  meets  the  church ’ [s]  specific  requirements | ` church`@149 (0.55) | 0.238 | 1.400 |
| `s` | `’` |  gap  in  the  state ’ [s]  surface  transportation | ` state`@84 (0.54) | 0.270 | 1.352 |
| `t` | `’` |  least ,  why  can ’ [t]  the  cans | ` can`@157 (0.42) | 0.359 | 1.343 |
| `s` | `’` |  of  making  a  man ’ [s]  self  the | ` man`@223 (0.60) | 0.244 | 1.331 |
| `s` | `’` |  infrastructure  will  grow  Ohio ’ [s]  economy , | ` Ohio`@92 (0.58) | 0.238 | 1.317 |
| `'t` | ` don` |  it ,  I  just  don ['t]  want  to | ` just`@196 (0.49) | 0.346 | 1.291 |

### Breakdown of non-prev attention targets
- Attends to **self**: 44 (53%)
- Attends to **BOS/position 0**: 0 (0%)
- Attends to **other position**: 39 (47%)

## Prev-Token Weight by Current Token (trained model)

Which tokens cause the head to attend LESS to the previous token?

### Tokens with LOWEST prev-token attention (weakest prev-token signal)

| Token | Mean Prev Weight | Count |
|-------|-----------------|-------|
| `t` | 0.3466 | 11 |
| `,”` | 0.3768 | 5 |
| `s` | 0.4333 | 24 |
| `.”` | 0.4998 | 6 |
| `.` | 0.5324 | 366 |
| `'t` | 0.5972 | 17 |
| `."` | 0.6065 | 6 |
| `).` | 0.6066 | 8 |
| ` when` | 0.6104 | 9 |
| ` points` | 0.6328 | 5 |
| `!` | 0.6502 | 14 |
| `),` | 0.6514 | 12 |
| ` our` | 0.6544 | 18 |
| ` because` | 0.6626 | 6 |
| ` interviews` | 0.6626 | 7 |
| ` medical` | 0.6688 | 5 |
| ` before` | 0.6755 | 7 |
| ` if` | 0.6771 | 13 |
| ` state` | 0.6788 | 6 |
| ` into` | 0.6817 | 6 |

### Tokens with HIGHEST prev-token attention (strongest prev-token signal)

| Token | Mean Prev Weight | Count |
|-------|-----------------|-------|
| ` NBA` | 0.9194 | 11 |
| `man` | 0.9212 | 7 |
| ` Fund` | 0.9330 | 8 |
| `arp` | 0.9392 | 6 |
| `ch` | 0.9403 | 6 |
| `aign` | 0.9405 | 5 |
| `aus` | 0.9450 | 8 |
| `F` | 0.9492 | 8 |
| ` Group` | 0.9495 | 7 |
| `�` | 0.9500 | 12 |
| `O` | 0.9542 | 6 |
| `W` | 0.9567 | 11 |
| `ue` | 0.9572 | 5 |
| `ON` | 0.9583 | 5 |
| `ana` | 0.9613 | 5 |
| `AN` | 0.9647 | 5 |
| `00` | 0.9658 | 11 |
| `30` | 0.9663 | 6 |
| `22` | 0.9785 | 8 |
| `N` | 0.9790 | 7 |