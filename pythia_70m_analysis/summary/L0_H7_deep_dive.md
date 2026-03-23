# L0_H7 Deep Dive

Analyzed 13101 positions across FineWeb validation docs.

## Attention Target Distribution

- **prev**: 5850 (44.7%)
- **self**: 3117 (23.8%)
- **near**: 2852 (21.8%)
- **distant**: 1282 (9.8%)

Prev weight: mean=0.259, median=0.142
Self weight: mean=0.121, median=0.052
Entropy: mean=2.295, median=2.558

## Behavior by Current Token

Which tokens cause which attention pattern?

### Most frequent tokens and their attention patterns

| Token | Count | Prev% | Self% | Near% | Dist% | Mean Prev Weight |
|-------|-------|-------|-------|-------|-------|-----------------|
| `,` | 586 | 5% | 78% | 16% | 1% | 0.041 |
| `.` | 457 | 4% | 71% | 25% | 1% | 0.041 |
| ` the` | 452 | 0% | 99% | 1% | 0% | 0.075 |
| ` and` | 282 | 6% | 78% | 13% | 3% | 0.068 |
| ` of` | 266 | 61% | 39% | 0% | 0% | 0.447 |
| ` to` | 252 | 42% | 56% | 2% | 0% | 0.287 |
| ` a` | 191 | 2% | 96% | 2% | 0% | 0.083 |
| ` in` | 169 | 43% | 17% | 39% | 1% | 0.212 |
| `-` | 108 | 58% | 42% | 0% | 0% | 0.492 |
| ` for` | 90 | 33% | 62% | 4% | 0% | 0.254 |
| ` is` | 84 | 18% | 24% | 38% | 20% | 0.108 |
| ` on` | 77 | 47% | 14% | 31% | 8% | 0.307 |
| ` I` | 77 | 31% | 0% | 52% | 17% | 0.119 |
| ` with` | 76 | 39% | 55% | 5% | 0% | 0.286 |
| ` that` | 75 | 37% | 61% | 1% | 0% | 0.332 |
| `:` | 75 | 7% | 63% | 28% | 3% | 0.118 |
| ` it` | 65 | 46% | 0% | 31% | 23% | 0.180 |
| ` have` | 65 | 72% | 9% | 18% | 0% | 0.553 |
| ` you` | 58 | 45% | 7% | 47% | 2% | 0.180 |
| ` -` | 57 | 23% | 23% | 51% | 4% | 0.096 |
| ` was` | 56 | 23% | 9% | 43% | 25% | 0.076 |
| ` (` | 52 | 29% | 19% | 42% | 10% | 0.106 |
| ` The` | 50 | 10% | 88% | 2% | 0% | 0.059 |
| ` as` | 49 | 61% | 0% | 29% | 10% | 0.326 |
| ` by` | 43 | 37% | 2% | 56% | 5% | 0.211 |
| `’` | 42 | 48% | 52% | 0% | 0% | 0.396 |
| ` are` | 42 | 40% | 55% | 5% | 0% | 0.244 |
| ` from` | 40 | 42% | 40% | 18% | 0% | 0.257 |
| ` be` | 39 | 62% | 31% | 8% | 0% | 0.429 |
| ` we` | 39 | 13% | 18% | 28% | 41% | 0.062 |
| ` not` | 37 | 81% | 0% | 11% | 8% | 0.395 |
| `s` | 35 | 80% | 0% | 20% | 0% | 0.378 |
| ` or` | 35 | 9% | 60% | 23% | 9% | 0.071 |
| ` an` | 33 | 0% | 97% | 3% | 0% | 0.093 |
| ` this` | 32 | 9% | 84% | 6% | 0% | 0.147 |
| ` your` | 31 | 0% | 81% | 19% | 0% | 0.060 |
| ` at` | 31 | 45% | 0% | 42% | 13% | 0.258 |
| ` will` | 30 | 40% | 0% | 47% | 13% | 0.241 |
| ` can` | 29 | 69% | 0% | 24% | 7% | 0.328 |
| ` but` | 28 | 7% | 0% | 46% | 46% | 0.087 |

### Tokens with highest SELF-attention rate (>30%, min 5 occurrences)

| Token | Count | Self% | Prev% | Mean Prev Wt | Example Context |
|-------|-------|-------|-------|-------------|-----------------|
| `"` | 9 | 100% | 0% | 0.014 |  " Good  Morning  America "  Wh |
| ` &` | 10 | 100% | 0% | 0.050 |  message  for  the  Heroes  &  Vill |
| `|` | 17 | 100% | 0% | 0.021 |  of  February  11 th |  | |
| `||` | 15 | 100% | 0% | 0.070 | |  | L il || Feb |
| ` using` | 8 | 100% | 0% | 0.067 |  should nt  have  been  using  that |
| `ana` | 5 | 100% | 0% | 0.112 |  September  24 .|  August ana  senior |
| ` the` | 452 | 99% | 0% | 0.075 |  was  so  surprising  when  the  Canadian |
| ` an` | 33 | 97% | 0% | 0.093 |  me ,  it  was  an  appropriate |
| ` a` | 191 | 96% | 2% | 0.083 |  the  Canadian  man  received  a  letter |
| `!` | 15 | 93% | 0% | 0.027 |  do  it  without  you !  Where |
| `”` | 13 | 92% | 8% | 0.103 |  “ certain  investigative  steps ”  in |
| `'` | 10 | 90% | 10% | 0.118 | ro ach  on  others '  territory |
| ` '` | 9 | 89% | 11% | 0.123 |  Wh istle - Stop  ' 08 |
| `)` | 27 | 89% | 0% | 0.042 |  ( by  crossing  boundaries )  doesn |
| ` The` | 50 | 88% | 10% | 0.059 |  other  government  benefits .  The  Manit |
| ` this` | 32 | 84% | 9% | 0.147 |  such  things .  Now  this  wouldn |
| ` your` | 31 | 81% | 0% | 0.060 |  rights  conversation  and  ensure  your  contribu |
| ` its` | 5 | 80% | 0% | 0.070 |  beyond  this ,  as  its  Har |
| ` and` | 282 | 78% | 6% | 0.068 | Ex haust ed  and  eup |
| `,` | 586 | 78% | 5% | 0.041 |  from  his  insurance  company ,  who |
| ` her` | 17 | 76% | 12% | 0.076 |  kinda  don 't  blame  her . |
| ` "` | 16 | 75% | 25% | 0.202 |  days  after  boarding  the  " Good |
| ` including` | 8 | 75% | 0% | 0.163 |  evaluation  and  validation ,  including  image |
| ` which` | 19 | 74% | 21% | 0.199 |  variety  of  matters ,  which  it |
| `).` | 11 | 73% | 9% | 0.015 | .  ( 18 66 ).  Crime |

### Tokens with highest PREV-TOKEN rate (>70%, min 5 occurrences)

| Token | Count | Prev% | Self% | Mean Prev Wt |
|-------|-------|-------|-------|-------------|
| `en` | 9 | 100% | 0% | 0.555 |
| ` why` | 5 | 100% | 0% | 0.336 |
| ` top` | 6 | 100% | 0% | 0.524 |
| ` dead` | 5 | 100% | 0% | 0.446 |
| `I` | 5 | 100% | 0% | 0.816 |
| `ic` | 5 | 100% | 0% | 0.932 |
| `C` | 6 | 100% | 0% | 0.959 |
| ` go` | 7 | 100% | 0% | 0.550 |
| `'t` | 17 | 100% | 0% | 0.666 |
| ` never` | 5 | 100% | 0% | 0.441 |
| ` am` | 11 | 100% | 0% | 0.807 |
| ` country` | 5 | 100% | 0% | 0.312 |
| ` help` | 5 | 100% | 0% | 0.630 |
| `y` | 8 | 100% | 0% | 0.733 |
| ` And` | 7 | 100% | 0% | 0.171 |
| `ro` | 6 | 100% | 0% | 0.949 |
| `by` | 7 | 100% | 0% | 0.561 |
| ` able` | 8 | 100% | 0% | 0.463 |
| `ue` | 5 | 100% | 0% | 0.516 |
| `ch` | 6 | 100% | 0% | 0.903 |
| ` though` | 5 | 100% | 0% | 0.256 |
| `ers` | 6 | 100% | 0% | 0.868 |
| `'ve` | 5 | 100% | 0% | 0.774 |
| `aus` | 8 | 100% | 0% | 0.915 |
| `the` | 6 | 100% | 0% | 0.916 |

## Behavior by Previous Token

Does the prev token determine the attention pattern?

### Previous tokens that cause HIGH self-attention in the NEXT position

| Prev Token | Count | Next Self% | Next Prev% | Mean Prev Wt |
|-----------|-------|-----------|-----------|-------------|
| `00` | 11 | 73% | 18% | 0.160 |
| ` others` | 11 | 73% | 0% | 0.080 |
| `S` | 11 | 73% | 27% | 0.263 |
| `F` | 11 | 64% | 0% | 0.036 |
| ` them` | 15 | 60% | 7% | 0.120 |
| ` get` | 12 | 58% | 25% | 0.227 |
| ` at` | 31 | 58% | 10% | 0.081 |
| `”` | 13 | 46% | 8% | 0.105 |
| ` over` | 11 | 45% | 36% | 0.247 |
| ` R` | 11 | 45% | 55% | 0.568 |
| ` one` | 20 | 45% | 30% | 0.269 |
| ` say` | 14 | 43% | 43% | 0.217 |
| ` all` | 28 | 43% | 7% | 0.129 |
| ` well` | 14 | 43% | 50% | 0.402 |
| ` people` | 12 | 42% | 8% | 0.058 |
| ` about` | 12 | 42% | 8% | 0.100 |
| ` work` | 12 | 42% | 50% | 0.311 |
| ` on` | 77 | 42% | 22% | 0.162 |
| `ing` | 17 | 41% | 12% | 0.047 |
| ` from` | 39 | 41% | 23% | 0.152 |

## Distant Attention Examples (delta > 5)

What makes the head look far back?

### Top 30 longest-range attention (sorted by distance)

| Token | Attended Token | Delta | Weight | Context |
|-------|---------------|-------|--------|---------|
| ` Liberty` | ` THE`@12 | 96 | 0.10 |  10  vols .  About  Liberty  Fund |
| ` He` | `en`@230 | 64 | 0.10 | ly  at  them .  He  held |
| ` she` | `,`@90 | 58 | 0.07 |  us .  As  if  she  invented |
| ` People` | `,`@86 | 56 | 0.11 |  in  Hep atitis  C  People  infected |
| ` nationally` | ` Vikings`@99 | 50 | 0.08 |  as  they  beat  four  nationally  ranked |
| ` Paul` | `The`@253 | 41 | 0.08 | - selling  ec ologist  Paul  E |
| ` Travis` | ` a`@302 | 40 | 0.09 | aver na  into  which  Travis  ( |
| ` Roy` | ` Ariz`@130 | 39 | 0.65 |  of  Anne  N .  Roy all |
| ` Hill` | ` a`@142 | 39 | 0.09 | ney  L us by  Hill , |
| ` runner` | `:`@88 | 38 | 0.12 | er  Bobby  Wagner  was  runner - |
| ` access` | ` disclaimer`@119 | 38 | 0.11 |  Micro  Center  has  no  access  to |
| ` 3` | `:`@38 | 38 | 0.12 | 18 66 ).  Part  3 , |
| ` trash` | `,`@54 | 38 | 0.14 |  stalls  for  eight  separate  trash  cans |
| ` Amsterdam` | `.,`@256 | 38 | 0.20 |  City ,  to  The  Amsterdam  News |
| ` Tomorrow` | ` of`@275 | 38 | 0.09 |  ham ster  forums !)  Tomorrow  we |
| `BA` | `|`@232 | 38 | 0.14 | : 00 || N BA  T |
| `BA` | `|`@246 | 38 | 0.14 | : 00 || N BA  T |
| `BA` | `|`@260 | 38 | 0.16 | : 00 || N BA  PLAY |
| `BA` | `|`@274 | 38 | 0.14 | : 30 || N BA  PLAY |
| `BA` | `|`@288 | 38 | 0.15 | : 30 || N BA  T |
| ` paid` | ` the`@61 | 38 | 0.10 |  not  need  to  be  paid  back |
| ` PDF` | `,`@249 | 38 | 0.08 |  reviews  are  published  in  PDF  format |
| ` Oliver` | ` a`@173 | 38 | 0.09 | t ano oga .  Oliver  has |
| ` Raz` | `),`@166 | 37 | 0.11 |  again .  Behind  him  Raz um |
| ` Side` | `),`@211 | 37 | 0.12 |  W inf oster .  Side  2 |
| ` 2013` | ` for`@132 | 37 | 0.13 |  -  LOT TER Y  2013 | |
| ` high` | ` and`@139 | 37 | 0.08 | ins  are  to  lower  high  cholesterol |
| ` U` | `,`@84 | 37 | 0.17 |  Shelby  Township  also  serves  U tica |
| ` Public` | `en`@38 | 36 | 0.15 |  terrible  if  Manit oba  Public  Insurance |
| ` Road` | `The`@112 | 36 | 0.10 |  at  365  Fan  Hill  Road . |

## Bigram Analysis: (prev, current) -> attention pattern

Do specific token PAIRS predict the attention pattern?

### Bigrams where current token attends to SELF (not prev)

| Prev -> Current | Count | Self% | Prev% |
|----------------|-------|-------|-------|
| ` of` -> ` the` | 61 | 100% | 0% |
| ` in` -> ` the` | 37 | 100% | 0% |
| ` to` -> ` the` | 28 | 100% | 0% |
| `,` -> ` the` | 21 | 100% | 0% |
| ` on` -> ` the` | 15 | 100% | 0% |
| ` and` -> ` the` | 14 | 100% | 0% |
| ` with` -> ` the` | 12 | 100% | 0% |
| ` from` -> ` the` | 10 | 100% | 0% |
| ` in` -> ` a` | 10 | 100% | 0% |
| ` for` -> ` the` | 10 | 100% | 0% |
| ` at` -> ` the` | 10 | 100% | 0% |
| `,` -> ` a` | 9 | 100% | 0% |
| ` by` -> ` the` | 8 | 100% | 0% |
| ` on` -> ` a` | 8 | 100% | 0% |
| `00` -> `||` | 8 | 100% | 0% |
| `,` -> ` to` | 7 | 100% | 0% |
| ` one` -> ` of` | 7 | 100% | 0% |
| ` as` -> ` a` | 7 | 100% | 0% |
| ` and` -> ` a` | 7 | 100% | 0% |
| ` of` -> ` a` | 6 | 100% | 0% |
| ` with` -> ` a` | 6 | 100% | 0% |
| ` to` -> ` a` | 6 | 100% | 0% |
| `IVE` -> `:` | 6 | 100% | 0% |
| ` Book` -> ` of` | 6 | 100% | 0% |
| ` have` -> ` a` | 5 | 100% | 0% |
| ` all` -> ` of` | 5 | 100% | 0% |
| `ch` -> `ly` | 5 | 100% | 0% |
| ` 18` -> `,` | 5 | 100% | 0% |
| ` Why` -> ` the` | 5 | 100% | 0% |
| `,` -> ` that` | 5 | 100% | 0% |