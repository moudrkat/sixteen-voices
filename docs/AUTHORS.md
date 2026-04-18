# Authors (77 total = 64 real + 13 synthetic)

All listed here match `data/authors/*.txt` at time of training. Real authors sourced from Project Gutenberg (script: `data/get_books.py`). Synthetic authors are hand-written text isolating one stylistic property.

---

## Real authors (64)

| Key | Source texts |
|---|---|
| **aesop** | Aesop's Fables |
| **african** | South African Folk-Tales · Anansi Stories |
| **alcott** | Little Women · Little Men |
| **andersen** | Fairy Tales of Hans Christian Andersen · Andersen's Fairy Tales (vol. 2) |
| **arabian** | The Arabian Nights Entertainments |
| **baker** | The Wonderful Stories of Fuz-Buz |
| **barrie** | Peter and Wendy |
| **baum** | Wizard of Oz (5 Oz books) |
| **blake** | Songs of Innocence and Experience |
| **brazilian** | Fairy Tales fr<br/>om Brazil |
| **browne** | Religio Medici |
| **burgess** | Peter Cottontail · Old Mother West Wind · Burgess Bird/Animal Book (4 books) |
| **burnett** | The Secret Garden · A Little Princess · Little Lord Fauntleroy |
| **byron** | Don Juan |
| **carlyle** | Sartor Resartus |
| **carroll** | Alice's Adventures in Wonderland · Through the Looking-Glass · Sylvie and Bruno |
| **chinese** | The Chinese Fairy Book · Myths and Legends of China |
| **collodi** | The Adventures of Pinocchio |
| **dequincey** | Confessions of an English Opium-Eater |
| **egyptian** | Legends of the Gods · Egyptian Tales (Petrie) |
| **filipino** | Filipino Popular Tales |
| **gibbon** | Decline and Fall of the Roman Empire |
| **grahame** | The Wind in the Willows |
| **greek_myth** | Myths of Greece and Rome |
| **grimm** | Grimm's Fairy Tales · Household Stories by the Brothers Grimm |
| **harris** | Uncle Remus · Nights with Uncle Remus |
| **hawthorne** | A Wonder-Book for Girls and Boys · Tanglewood Tales |
| **homer** | The Iliad (Lang prose) · The Odyssey (Butcher-Lang prose) |
| **indian** | Indian Fairy Tales · Jataka Tales · The Magic Bed |
| **italian** | Italian Popular Tales |
| **jacobs** | English / Celtic / Indian Fairy Tales |
| **japanese** | Japanese Fairy Tales |
| **johnson** | Rasselas |
| **kingsley** | The Water-Babies |
| **kipling** | The Jungle Book · The Second Jungle Book · Just So Stories |
| **korean** | Korean Folk Tales |
| **lang** | 12 Andrew-Lang Fairy Books (Blue, Red, Green, Yellow, ..., Lilac) |
| **lear** | A Book of Nonsense · Nonsense Songs, Stories, Botany, and Alphabets |
| **lofting** | Doctor Dolittle books |
| **lovecraft** | The Call of Cthulhu · At the Mountains of Madness · Charles Dexter Ward |
| **maeterlinck** | The Blue Bird for Children |
| **maya** | Popol Vuh |
| **melville** | Moby Dick |
| **milton** | Paradise Lost |
| **montgomery** | Anne of Green Gables · Anne of Avonlea |
| **nesbit** | The Enchanted Castle · Five Children and It · The Book of Dragons · others |
| **norse** | East o' the Sun and West o' the Moon · Norse Mythology |
| **pater** | The Renaissance |
| **poe** | The Raven and Other Poems · The Works of Edgar Allan Poe, vol. 1 |
| **pyle** | The Merry Adventures of Robin Hood · King Arthur · Pepper & Salt |
| **quiroga** | South American Jungle Tales |
| **ruskin** | The King of the Golden River |
| **russian** | Russian Fairy Tales |
| **sewell** | Black Beauty |
| **shelley** | Frankenstein |
| **spyri** | Heidi |
| **stevenson** | Treasure Island · Kidnapped |
| **stoker** | Dracula |
| **tennyson** | Idylls of the King |
| **twain** | Tom Sawyer · Huckleberry Finn · The Prince and the Pauper |
| **verne** | Twenty Thousand Leagues Under the Seas · Around the World in Eighty Days |
| **wells** | The Time Machine · The War of the Worlds · The Invisible Man |
| **wilde** | The Happy Prince and Other Tales · A House of Pomegranates |
| **wyss** | The Swiss Family Robinson |

---

## Synthetic style archetypes (13)

Each was hand-written (or lightly generated and curated) to isolate one property, so that SAE features correlating with that archetype's text map cleanly to a single stylistic axis. Without these, it's hard to tell what a feature is actually detecting.

| Key | Stylistic axis isolated |
|---|---|
| **minimalist** | Very short, bare sentences. "A cat sat. It saw a bird. The bird flew." |
| **dialogue** | All conversation, quote-and-attribution. "'What do you know?' asked the moon." |
| **poet** | Line breaks, rhythm, stanza-like structure |
| **cozy** | Food, warmth, domestic comfort (kitchen, cinnamon, honey, soup) |
| **dark** | Ominous atmosphere, uncanny negation |
| **firstperson** | "I" narration, inside view |
| **questioner** | Frequent interrogatives — "Why? How? What if?" |
| **fabulist** | Moral-fable rhythm, animal characters, aphorism endings |
| **simple_vocab** | Short, everyday words only |
| **unusual_vocab** | Rare, Latinate, elevated register vocabulary |
| **reporter** | Neutral third-person reporting voice |
| **rambler** | Very long sentences, discursive, many clauses |
| **repeater** | Anaphora / repetition patterns |

---

## A note on counts

An earlier draft of the poster said "69 real + 8 synthetic." The actual numbers ended up being **64 real + 13 synthetic** after cleanup. The final total — 77 — is unchanged.

### Authors that were in the download list but did not make the final training set

Listed in `data/get_books.py → BOOKS` but no corresponding `data/authors/*.txt` file exists:

`dunsany, macdonald, native_american, perrault, potter, turkish, yeats`

These were **dropped due to contamination** — overlap with the TinyStories training distribution, corrupted downloads, or Project Gutenberg boilerplate that survived cleaning. Rather than risk polluting the adapter-per-author comparison, they were excluded.

(If anyone asks at the session: *"We downloaded ~71 authors but dropped 7 for contamination, ending at 64 real + 13 synthetic = 77."*)

---

## Training vs evaluation texts

Two separate directories:

- `data/authors/*.txt` — **training text**. What the LoRA adapters fit on.
- `data/eval/*.txt` — **held-out evaluation text**, extracted cleanly from the same author corpus but disjoint from the training chunks. This is what perplexity is measured on for head knockout, steering, and all other experiments.

The eval texts were extracted once, verified clean (no boilerplate, no TinyStories-style interference), and then frozen. The contaminated/dropped authors above don't appear in `data/eval/` either.

Code: `src/sixteen_voices/text.py → load_eval_text`. Extraction script: `scripts/extract_prose_texts.py`.
