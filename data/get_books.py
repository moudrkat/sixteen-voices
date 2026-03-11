#!/usr/bin/env python3
"""Download individual books from Project Gutenberg.

Saves one file per book to data/stories/{book_key}.txt.

Usage:
    python data/get_books.py              # download all
    python data/get_books.py --list       # list books
    python data/get_books.py --only grimm_tales alice
"""

import argparse
import time
from pathlib import Path

from gutenberg import fetch_text, strip_gutenberg

STORIES_DIR = Path(__file__).parent / "stories"

# (gutenberg_id, title, author_key)
BOOKS = {
    # Lang Fairy Books
    "lang_blue":       (503,   "The Blue Fairy Book",          "lang"),
    "lang_red":        (640,   "The Red Fairy Book",           "lang"),
    "lang_green":      (2707,  "The Green Fairy Book",         "lang"),
    "lang_yellow":     (641,   "The Yellow Fairy Book",        "lang"),
    "lang_pink":       (5765,  "The Pink Fairy Book",          "lang"),
    "lang_grey":       (2710,  "The Grey Fairy Book",          "lang"),
    "lang_violet":     (10557, "The Violet Fairy Book",        "lang"),
    "lang_crimson":    (2435,  "The Crimson Fairy Book",       "lang"),
    "lang_brown":      (3170,  "The Brown Fairy Book",         "lang"),
    "lang_orange":     (3027,  "The Orange Fairy Book",        "lang"),
    "lang_olive":      (4913,  "The Olive Fairy Book",         "lang"),
    "lang_lilac":      (9659,  "The Lilac Fairy Book",         "lang"),
    # Grimm
    "grimm_tales":     (2591,  "Grimm's Fairy Tales",                    "grimm"),
    "grimm_household": (5314,  "Household Stories by the Brothers Grimm", "grimm"),
    # Andersen
    "andersen_tales":  (1597,  "Fairy Tales of Hans Christian Andersen",  "andersen"),
    "andersen_vol2":   (27200, "Andersen's Fairy Tales (vol. 2)",         "andersen"),
    # Perrault
    "perrault":        (2738,  "The Fairy Tales of Charles Perrault",     "perrault"),
    # Collodi
    "pinocchio":       (500,   "The Adventures of Pinocchio",            "collodi"),
    # Carroll
    "alice":           (11,    "Alice's Adventures in Wonderland",        "carroll"),
    "looking_glass":   (12,    "Through the Looking-Glass",              "carroll"),
    "sylvie_bruno":    (620,   "Sylvie and Bruno",                       "carroll"),
    # Wilde
    "happy_prince":    (773,   "The Happy Prince and Other Tales",       "wilde"),
    "pomegranates":    (902,   "A House of Pomegranates",                "wilde"),
    # Kipling
    "jungle_book":     (70,    "The Jungle Book",                        "kipling"),
    "jungle_book2":    (236,   "The Second Jungle Book",                 "kipling"),
    "just_so":         (2850,  "Just So Stories",                        "kipling"),
    # Aesop
    "aesop":           (21,    "Aesop's Fables",                         "aesop"),
    # MacDonald
    "princess_goblin": (25,    "The Princess and the Goblin",            "macdonald"),
    "north_wind":      (1327,  "At the Back of the North Wind",         "macdonald"),
    "princess_curdie": (709,   "The Princess and Curdie",               "macdonald"),
    # Potter
    "peter_rabbit":    (14838, "The Tale of Peter Rabbit",              "potter"),
    "gloucester":      (14837, "The Tailor of Gloucester",              "potter"),
    "squirrel_nutkin": (15156, "The Tale of Squirrel Nutkin",           "potter"),
    "benjamin_bunny":  (14847, "The Tale of Benjamin Bunny",            "potter"),
    "two_bad_mice":    (14407, "The Tale of Two Bad Mice",              "potter"),
    "mrs_tiggywinkle": (14814, "The Tale of Mrs. Tiggy-Winkle",        "potter"),
    # Baum
    "wizard_oz":       (55,    "The Wonderful Wizard of Oz",            "baum"),
    "land_oz":         (54,    "The Marvelous Land of Oz",              "baum"),
    "emerald_city":    (517,   "The Emerald City of Oz",                "baum"),
    "ozma_oz":         (486,   "Ozma of Oz",                            "baum"),
    "road_oz":         (26624, "The Road to Oz",                        "baum"),
    # Nesbit
    "enchanted_castle": (778,  "The Enchanted Castle",                  "nesbit"),
    "five_children":   (23661, "Five Children and It",                   "nesbit"),
    "book_dragons":    (6234,  "The Book of Dragons",                   "nesbit"),
    "railway_children": (1874, "The Railway Children",                  "nesbit"),
    "wonderful_garden": (3015, "The Wonderful Garden",                  "nesbit"),
    # Barrie
    "peter_wendy":     (16,    "Peter and Wendy",                       "barrie"),
    # Grahame
    "wind_willows":    (289,   "The Wind in the Willows",               "grahame"),
    # Jacobs
    "english_fairy":   (7439,  "English Fairy Tales",                    "jacobs"),
    "celtic_fairy":    (14367, "Celtic Fairy Tales",                     "jacobs"),
    "indian_fairy":    (7128,  "Indian Fairy Tales",                     "jacobs"),
    # Hawthorne
    "wonder_book":     (9251,  "A Wonder-Book for Girls and Boys",       "hawthorne"),
    "tanglewood":      (9259,  "Tanglewood Tales",                      "hawthorne"),
    # Burnett
    "secret_garden":   (113,   "The Secret Garden",                     "burnett"),
    "little_princess": (146,   "A Little Princess",                     "burnett"),
    "little_lord":     (479,   "Little Lord Fauntleroy",                "burnett"),
    # Alcott
    "little_women":    (514,   "Little Women",                          "alcott"),
    "little_men":      (2788,  "Little Men",                            "alcott"),
    # Spyri
    "heidi":           (1448,  "Heidi",                                 "spyri"),
    # Stevenson
    "treasure_island": (120,   "Treasure Island",                       "stevenson"),
    "kidnapped":       (421,   "Kidnapped",                             "stevenson"),
    # Twain
    "tom_sawyer":      (74,    "The Adventures of Tom Sawyer",          "twain"),
    "huck_finn":       (76,    "Adventures of Huckleberry Finn",        "twain"),
    "prince_pauper":   (1837,  "The Prince and the Pauper",             "twain"),
    # Sewell
    "black_beauty":    (271,   "Black Beauty",                          "sewell"),
    # Montgomery
    "anne_green":      (45,    "Anne of Green Gables",                  "montgomery"),
    "anne_avonlea":    (47,    "Anne of Avonlea",                       "montgomery"),
    # Lofting
    "dr_dolittle":     (501,   "The Story of Doctor Dolittle",          "lofting"),
    "voyages_dolittle": (1154, "The Voyages of Doctor Dolittle",        "lofting"),
    # Burgess
    "peter_cottontail": (4980, "The Adventures of Peter Cottontail",    "burgess"),
    "mother_west_wind": (4979, "Old Mother West Wind",                  "burgess"),
    "burgess_bird":    (6567,  "The Burgess Bird Book for Children",    "burgess"),
    "burgess_animal":  (11559, "The Burgess Animal Book for Children",  "burgess"),
    # Lear
    "book_nonsense":   (13650, "A Book of Nonsense",                    "lear"),
    "nonsense_songs":  (13649, "Nonsense Songs, Stories, Botany, and Alphabets", "lear"),
    # Pyle
    "robin_hood":      (964,   "The Merry Adventures of Robin Hood",    "pyle"),
    "king_arthur":     (5712,  "The Story of King Arthur and His Knights", "pyle"),
    "pepper_salt":     (3088,  "Pepper & Salt",                         "pyle"),
    # Cultural collections
    "arabian_nights":  (2814,  "The Arabian Nights Entertainments",     "arabian"),
    "japanese_fairy":  (4018,  "Japanese Fairy Tales",                   "japanese"),
    "russian_fairy":   (12851, "Russian Fairy Tales",                    "russian"),
    "norse_tales":     (8653,  "East o' the Sun and West o' the Moon",  "norse"),
    "italian_fairy":   (23634, "Italian Popular Tales",                 "italian"),
    "turkish_fairy":   (31061, "Turkish Fairy Tales",                   "turkish"),
    # Additional children's classics
    "water_babies":    (1018,  "The Water-Babies",                      "kingsley"),
    "blue_bird":       (8606,  "The Blue Bird for Children",            "maeterlinck"),
    "swiss_family":    (3249,  "The Swiss Family Robinson",             "wyss"),
    # African & diaspora
    "african_folk":    (38339, "South African Folk-Tales",              "african"),
    "anansi":          (28944, "Anansi Stories",                        "african"),
    "uncle_remus":     (2306,  "Uncle Remus",                           "harris"),
    "nights_uncle_remus": (6086, "Nights with Uncle Remus",             "harris"),
    # Asian traditions
    "chinese_fairy":   (29939, "The Chinese Fairy Book",                "chinese"),
    "chinese_myths":   (15250, "Myths and Legends of China",           "chinese"),
    "filipino_tales":  (12814, "Filipino Popular Tales",                "filipino"),
    "korean_tales":    (51002, "Korean Folk Tales",                     "korean"),
    "indian_nights":   (31209, "Indian Fairy Tales",                    "indian"),
    "jataka_tales":    (62514, "Jataka Tales",                          "indian"),
    "magic_bed":       (37708, "The Magic Bed: East Indian Fairy-Tales","indian"),
    # Americas
    "brazilian_tales": (24714, "Fairy Tales from Brazil",               "brazilian"),
    "native_american": (22495, "Myths of the North American Indians",  "native_american"),
    "maya_quiche":     (56550, "Popol Vuh",                             "maya"),
    # Ancient mythology
    "greek_myths":     (22381, "Myths of Greece and Rome",             "greek_myth"),
    "norse_myths":     (28497, "Norse Mythology",                       "norse"),
    "egyptian_myths":  (9411,  "Legends of the Gods",                  "egyptian"),
    "egyptian_tales":  (7386,  "Egyptian Tales (Petrie)",               "egyptian"),
    "celtic_myths":    (14727, "The Celtic Twilight",                   "yeats"),
    "iliad_lang":      (6130,  "The Iliad (Lang prose)",               "homer"),
    "odyssey_lang":    (1728,  "The Odyssey (Butcher-Lang prose)",     "homer"),
    # Gothic / early sci-fi
    "frankenstein":    (84,    "Frankenstein",                          "shelley"),
    "dracula":         (345,   "Dracula",                               "stoker"),
    "time_machine":    (35,    "The Time Machine",                      "wells"),
    "war_worlds":      (36,    "The War of the Worlds",                 "wells"),
    "invisible_man":   (5230,  "The Invisible Man",                     "wells"),
    "twenty_thousand": (164,   "Twenty Thousand Leagues Under the Seas", "verne"),
    "around_world":    (103,   "Around the World in Eighty Days",       "verne"),
    # Poetry / verse
    "songs_innocence": (1934,  "Songs of Innocence and Experience",    "blake"),
    "raven_poe":       (17192, "The Raven and Other Poems",            "poe"),
    "poe_tales":       (2147,  "The Works of Edgar Allan Poe, vol. 1", "poe"),
    # More diverse novelists
    "jungle_tales":    (46051, "South American Jungle Tales",          "quiroga"),
    "wonderful_stories": (36308, "The Wonderful Stories of Fuz-Buz",   "baker"),
    "tales_wonder":    (29661, "Tales of Wonder",                       "dunsany"),
    "book_wonder":     (7477,  "The Book of Wonder",                    "dunsany"),
    "king_golden":     (701,   "The King of the Golden River",         "ruskin"),
    # Dense/archaic vocabulary (OOD from TinyStories)
    "cthulhu":         (68283, "The Call of Cthulhu",                    "lovecraft"),
    "mountains_madness": (70652, "At the Mountains of Madness",          "lovecraft"),
    "charles_ward":    (73547, "The Case of Charles Dexter Ward",        "lovecraft"),
    "moby_dick":       (2701,  "Moby Dick",                             "melville"),
    "paradise_lost":   (20,    "Paradise Lost",                          "milton"),
    "sartor_resartus": (1051,  "Sartor Resartus",                        "carlyle"),
    "religio_medici":  (586,   "Religio Medici",                         "browne"),
    "opium_eater":     (2040,  "Confessions of an English Opium-Eater",  "dequincey"),
    "decline_fall":    (25717, "Decline and Fall of the Roman Empire",    "gibbon"),
    "rasselas":        (652,   "Rasselas",                               "johnson"),
    "idylls_king":     (610,   "Idylls of the King",                     "tennyson"),
    "don_juan":        (21700, "Don Juan",                               "byron"),
    "renaissance":     (2398,  "The Renaissance",                        "pater"),
}


def download_book(key: str, book_id: int, title: str, *, force: bool = False) -> bool:
    out_path = STORIES_DIR / f"{key}.txt"
    if out_path.exists() and not force:
        print(f"  [{key}] already exists -- skip")
        return True

    print(f"  [{key}] fetching: {title} (id={book_id})...", end=" ", flush=True)
    raw = fetch_text(book_id)
    if raw is None:
        print("FAILED")
        return False

    clean = strip_gutenberg(raw)
    words = len(clean.split())
    if words < 100:
        print(f"too short ({words} words) -- skip")
        return False

    out_path.write_text(clean, encoding="utf-8")
    print(f"{words:,} words")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download books from Project Gutenberg")
    parser.add_argument("--only", nargs="+", metavar="KEY")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\n{len(BOOKS)} books available:\n")
        for key, (book_id, title, author) in sorted(BOOKS.items()):
            exists = (STORIES_DIR / f"{key}.txt").exists()
            mark = "[x]" if exists else "[ ]"
            print(f"  {mark} {key:25s}  {author:16s}  {book_id:>6d}  {title}")
        return

    STORIES_DIR.mkdir(parents=True, exist_ok=True)
    targets = {k: BOOKS[k] for k in (args.only or BOOKS)}
    if args.only:
        bad = [k for k in args.only if k not in BOOKS]
        if bad:
            print(f"Unknown: {', '.join(bad)}. Use --list.")
            return

    print(f"Downloading {len(targets)} books...\n")
    ok, fail = 0, 0
    for i, (key, (book_id, title, _)) in enumerate(targets.items()):
        if download_book(key, book_id, title, force=args.force):
            ok += 1
        else:
            fail += 1
        if i < len(targets) - 1:
            time.sleep(2)

    print(f"\nDone: {ok} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
