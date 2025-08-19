import os
import re
import sys
import requests
import mimetypes
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Union
from datetime import datetime
import PyPDF2
from io import BytesIO
from paths import DATA_DIR  # <-- Import your centralized data directory path


def save_markdown(title: str, content: str, output_dir: Path):
    """Save content as a markdown file."""
    safe_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title)[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{timestamp}.md"
    filepath = output_dir / filename

    md_content = f"# {title}\n\n" + content
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ Saved: {filepath}")


def scrape_html(url: str, headers: dict, output_dir: Path):
    """Scrape HTML page and save content."""
    response = requests.get(url, headers=headers, timeout=100)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else "untitled"

    article_tag = soup.find("article")
    if article_tag:
        content = "\n".join(p.get_text(strip=True) for p in article_tag.find_all("p"))
    else:
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text(strip=True) for p in paragraphs)

    if not content.strip():
        print(f"⚠️ No readable content found for {url}. Skipping...")
        return

    save_markdown(title_text, content, output_dir)


def scrape_pdf(url: str, headers: dict, output_dir: Path):
    """Download and extract text from a PDF."""
    response = requests.get(url, headers=headers, timeout=100)
    response.raise_for_status()

    pdf_stream = BytesIO(response.content)
    reader = PyPDF2.PdfReader(pdf_stream)

    all_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            all_text.append(f"## Page {page_num}\n\n{text.strip()}")

    if not all_text:
        print(f"⚠️ No text extracted from PDF: {url}")
        return

    title_text = url.split("/")[-1] or "downloaded_pdf"
    save_markdown(title_text, "\n\n".join(all_text), output_dir)


def scrape_and_save_articles(
    urls: Union[str, List[str]],
    output_dir: Path = Path(DATA_DIR),  # Use centralized DATA_DIR here
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
) -> None:
    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list) or not urls:
        raise ValueError("`urls` must be a non-empty string or list of strings.")

    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent}

    for url in urls:
        try:
            print(f"🔍 Scraping: {url}")

            mime_type, _ = mimetypes.guess_type(url)
            if mime_type == "application/pdf" or url.lower().endswith(".pdf"):
                scrape_pdf(url, headers, output_dir)
            else:
                scrape_html(url, headers, output_dir)

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error for {url}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error for {url}: {e}")


if __name__ == "__main__":
    try:
        scrape_and_save_articles([
            # "https://kidshealth.org/en/parents/feed-child-athlete.html",
            # "https://nextlevelathleticsusa.com/athlete-training/best-youth-athlete-training-exercises/",
            # "https://memphisyouthathletics.org/what-recovery-techniques-are-best-for-young-athletes/",
            # "https://appliedsportpsych.org/resources/resources-for-parents/dos-and-donts-for-parents-of-young-athletes/",
            "https://www.stlouischildrens.org/sites/legacy/files/2020-05/383626_SLC%20YAC%20Strength%20Training%20Exercise%20Program%20Booklet_WEB.pdf",
            'https://www.sportsdietitians.com.au/wp-content/uploads/2015/04/SDA_Junior-Athlete_FINAL.pdf',
            'https://youthsports.rutgers.edu/wp-content/uploads/Guidelines-for-Supportive-Parents.pdf'
        ])
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)