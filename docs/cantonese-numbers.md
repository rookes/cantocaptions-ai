# Which numbers get converted

Rules for `languages/yue/numbers.py` (Chinese numeral → Arabic).


A written-out Chinese numeral is often the *better* subtitle, so conversion is opt-in per
case rather than a blanket rewrite. Three contexts convert unconditionally, because there
the standard numeral is the convention whatever the magnitude:

| context | example |
|---|---|
| a full clock time, hour ≤ 23 and minute ≤ 59 | 八點三十分 → 8點30分 |
| a date or a numeric name (`月`, `號`) | 一月二十三號 → 1月23號, 五號 → 5號 |
| a year written digit-by-digit | 一二三四年 → 1234年 |

Everything else is a plain quantity, and converts only if **all** of these hold:

1. it uses a multiplicity quantifier — 八 and 一二三 stay as they are;
2. no hedge sits on either side of it — 幾十個, 十幾個, 四十幾 are approximations;
3. it uses no colloquial ten (廿, 卅) — that register does not survive "21";
4. it is not a 萬/億 covering the whole number — where the largest quantifier is also the
   *last* character, the Chinese form already **is** the compact one: 二十五萬 beats
   250,000, and 一百二十七億 beats 12,700,000,000. A digit after it flips this, because
   then the digits are the shorter form again: 五十五萬一千 → 551,000;
5. it writes at least one digit — a lone 十 or 百 is a quantifier, not a number, and
   converting it turns 十分好 into 10分好.

Converted values ≥ 10,000 are grouped with **half-width** commas (45,010). That survives
cleaning because `punctuation.toml`'s `,` → `，` rule runs *before* the `chinese_numbers`
step in `rules/pipeline.toml` and nothing after it touches half-width commas — an
important ordering constraint for a custom `--clean_rules_dir`.

Two traps in the parsing itself:

- **A big unit multiplies everything written in front of it**, not just the digit beside
  it. 二十五萬 is (20 + 5) × 10,000 = 250,000; accumulating left to right and applying 萬
  to the 五 alone gives 50,020. Digits accumulate into a *section* that the next 萬/億
  multiplies whole.
- **A trailing bare digit takes the place below the last unit written.** 三百五 is 350 and
  五十五萬一 is 551,000 — unless a 零 marked the ones place explicitly, which is what makes
  五十五萬零一 550,001 and 一百零八 108.

**`點` is a point and a degree as well as an hour**, so the whole `點` construct is judged
at once and both halves share the verdict — a numeral next to a `點` is never converted by
the general rules above.

A numeral on *both* sides fixes the reading, and both convert whatever their size: 八點三十分
→ `8點30分`, 八點六十分 → `8點60分`, 三十六點五度 → `36點5度`, 三點五 → `3點5`. Where the pair
could genuinely be a clock time — a following `分`, hour in `MIN_HOUR..MAX_HOUR`, minute
≤ `MAX_MINUTE` — the minute is zero-padded to read as a clock face (八點零五分 → `8點05分`).
`MIN_HOUR` is 1 rather than 0 because 零點五分 is the decimal 0.5, not five past midnight.

With nothing after the `點` the reading is not fixed, so the numeral converts only once it
is too large to be an hour: above `MAX_BARE_HOUR` (12), a bare hour being written on the
12-hour clock. 五點 and 十二點 stay; 十九點 → `19點`, 一百點 → `100點`.

Three things suppress the construct entirely, neither half converting: a hedge on either
side (十九點幾, 十幾點, 十二點幾分鐘), a colloquial ten (廿三點), and the quarter-hour count
`個字` (十點四個字), where the numeral counts fifths of an hour rather than minutes.

