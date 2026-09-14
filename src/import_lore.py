import os
import re
import requests
from bs4 import BeautifulSoup


OUTPUT_FOLDER = "data/raw"

ARTICLES = [
    "Aragorn",
    "Arwen",
    "Gandalf",
    "Frodo Baggins",
    "Samwise Gamgee",
    "Bilbo Baggins",
    "Meriadoc Brandybuck",
    "Peregrin Took",
    "Boromir",
    "Faramir",
    "Denethor II",
    "Théoden",
    "Éomer",
    "Éowyn",
    "Legolas",
    "Gimli",
    "Galadriel",
    "Celeborn",
    "Elrond",
    "Saruman",
    "Radagast",
    "Sauron",
    "Morgoth",
    "Gollum",
    "Treebeard",
    "Tom Bombadil",

    "Fëanor",
    "Fingolfin",
    "Finarfin",
    "Finrod",
    "Beren",
    "Lúthien",
    "Túrin",
    "Húrin",
    "Thingol",
    "Melian",
    "Eärendil",
    "Maedhros",

    "Elendil",
    "Isildur",
    "Anárion",
    "Gil-galad",
    "Celebrimbor",
    "Ar-Pharazôn",
    "Elros",

    "Ainur",
    "Valar",
    "Maiar",
    "Istari",
    "Elves",
    "Noldor",
    "Sindar",
    "Dúnedain",
    "Númenóreans",
    "Dwarves",
    "Hobbits",
    "Orcs",
    "Ents",
    "Nazgûl",

    "Middle-earth",
    "Valinor",
    "Beleriand",
    "Númenor",
    "Gondor",
    "Arnor",
    "Rohan",
    "Mordor",
    "Shire",
    "Rivendell",
    "Lothlórien",
    "Moria",
    "Erebor",
    "Minas Tirith",
    "Osgiliath",
    "Mount Doom",
    "Barad-dûr",
    "Isengard",
    "Grey Havens",

    "One Ring",
    "Rings of Power",
    "Narya",
    "Nenya",
    "Vilya",
    "Silmarils",
    "Palantíri",
    "Narsil",
    "Andúril",

    "War of the Ring",
    "War of the Last Alliance",
    "Battle of the Pelennor Fields",
    "Battle of Helm's Deep",
    "Council of Elrond",
    "Quest of Erebor",
    "Fall of Númenor",
    "War of Wrath",
    "Kinslaying",

    "First Age",
    "Second Age",
    "Third Age",
    "Fourth Age",
    "Two Trees of Valinor",
]


UNWANTED_SECTIONS = [
    "adaptations",
    "portrayal",
    "video games",
    "games",
    "film",
    "television",
    "radio series",
    "merchandise",
    "behind the scenes",
    "references",
    "external links",
    "see also",
    "bibliography",
    "notes",
    "other versions of the legendarium",
    "inspiration",
]

def safe_filename(title):
    filename = title.lower()

    replacements = {
        "ë": "e",
        "é": "e",
        "á": "a",
        "ú": "u",
        "ó": "o",
        "í": "i",
        "ä": "a",
        "ö": "o",
        "û": "u",
        "ô": "o",
    }

    for old, new in replacements.items():
        filename = filename.replace(old, new)

    filename = re.sub(r"[^a-z0-9]+", "_", filename)

    return filename.strip("_") + ".txt"


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def section_is_unwanted(section_title):
    section_lower = section_title.lower()

    return any(
        unwanted in section_lower
        for unwanted in UNWANTED_SECTIONS
    )


def download_article(title):
    url_title = title.replace(" ", "_")
    url = f"https://tolkiengateway.net/wiki/{url_title}"

    print(f"Downloading: {title}")

    headers = {
        "User-Agent": "Loremaster educational RAG project"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    content = soup.select_one(".mw-parser-output")

    if content is None:
        print(f"  Could not find content: {title}")
        return

    # Remove obvious webpage clutter
    for unwanted in content.select(
        "table, style, script, .navbox, .toc, "
        ".mw-editsection, .references, .reflist, "
        "sup.reference, figure"
    ):
        unwanted.decompose()

    sections = []

    current_section = "Introduction"
    current_parts = []

    # Walk through headings and actual prose in document order
    elements = content.find_all(
        ["h2", "h3", "p", "ul", "ol"],
        recursive=True
    )

    for element in elements:

        # ------------------------------------------
        # New section heading
        # ------------------------------------------
        if element.name in ["h2", "h3"]:

            previous_text = clean_text(
                " ".join(current_parts)
            )

            if (
                previous_text
                and not section_is_unwanted(current_section)
            ):
                sections.append(
                    (current_section, previous_text)
                )

            current_section = clean_text(
                element.get_text(" ", strip=True)
            )

            current_parts = []

            continue

        # ------------------------------------------
        # Ignore unwanted sections entirely
        # ------------------------------------------
        if section_is_unwanted(current_section):
            continue

        # ------------------------------------------
        # Save normal article prose
        # ------------------------------------------
        text = clean_text(
            element.get_text(" ", strip=True)
        )

        if text:
            current_parts.append(text)

    # Save final section
    final_text = clean_text(
        " ".join(current_parts)
    )

    if (
        final_text
        and not section_is_unwanted(current_section)
    ):
        sections.append(
            (current_section, final_text)
        )

    if not sections:
        print(f"  No usable sections found: {title}")
        return

    filename = safe_filename(title)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(f"TITLE: {title}\n")
        file.write(f"SOURCE: {response.url}\n")
        file.write("SOURCE SITE: Tolkien Gateway\n")
        file.write("LICENSE: CC BY-SA 4.0\n")

        for section_title, section_text in sections:

            file.write("\n")
            file.write(
                f"## SECTION: {section_title}\n"
            )
            file.write(section_text)
            file.write("\n")

    print(f"  Saved {len(sections)} sections: {output_path}")

def main():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    successful = 0
    failed = 0

    for article in ARTICLES:

        try:
            download_article(article)
            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"  ERROR downloading {article}: "
                f"{error}"
            )

    print("\n================================")
    print("LORE IMPORT COMPLETE")
    print("================================")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")


if __name__ == "__main__":
    main()