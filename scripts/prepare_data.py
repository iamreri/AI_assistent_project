# -*- coding: utf-8 -*-
"""
Подготовка базы знаний МГТУ им. Г.И. Носова для RAG.

Этапы:
  1. Извлечение текста из PDF (PyMuPDF) / DOCX (python-docx) / TXT / MD.
  2. Очистка: мусор, переносы, кавычки, тире, лишние пробелы.
  3. Дедупликация документов (SHA-256 нормализованного текста).
  4. Чанкинг двумя вариантами: fixed и semantic.
  5. Сохранение чанков в JSONL с метаданными.

Запуск:
  python scripts/prepare_data.py \
      --index data/documents/docs_index.csv \
      --raw-dir data/documents/raw \
      --out-dir data
"""

import argparse
import bisect
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

CHUNK_MAX_CHARS = 1600   # ~500 токенов (для русского 1 токен ~ 3 символа)
CHUNK_MIN_CHARS = 400
CHUNK_OVERLAP_CHARS = 200

LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
PAGE_NUM_RE = re.compile(r"^\s*(стр(?:аница)?\.?\s*)?\d+\s*(из\s*\d+)?\s*$", re.IGNORECASE)
HEADING_MD_RE = re.compile(r"^\s*(#{1,6})\s+")
HEADING_NUM_RE = re.compile(r"^\s*\d+\.\s+\S")
HEADING_WORD_RE = re.compile(r"^\s*(Раздел|ГЛАВА|Глава|Приложение|Статья|ПРИКАЗЫВАЮ)\b", re.IGNORECASE)

TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ы": "y", "э": "e", "ю": "yu", "я": "ya", "ъ": "", "ь": ""})


# ------------------------------------------------------------- извлечение
def extract_pdf(path):
    import fitz
    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def extract_docx(path):
    import docx
    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs]
    for t in d.tables:
        rows = [" | ".join(c.text.strip().replace("\n", " ") for c in r.cells) for r in t.rows]
        paras.append("[ТАБЛИЦА]\n" + "\n".join(rows))
    return ["\n\n".join(paras)]


def extract_plain(path):
    return [path.read_text(encoding="utf-8", errors="ignore")]


def extract(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in (".docx", ".doc"):
        return extract_docx(path)
    return extract_plain(path)


# ------------------------------------------------------------- очистка
def clean_text(text):
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "").replace("\t", " ")

    # склейка слов, разорванных переносом
    text = re.sub(r"([а-яёa-z])-\n([а-яёa-z])", r"\1\2", text, flags=re.IGNORECASE)

    lines = []
    for line in text.split("\n"):
        s = line.rstrip()
        if PAGE_NUM_RE.match(s.strip()):          # номера страниц и «стр. X из Y»
            continue
        s = s.replace("\u201c", "\u00ab").replace("\u201d", "\u00bb")
        s = s.replace("\u2018", "\u00ab").replace("\u2019", "\u00bb")
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_hash(text):
    s = text.lower().replace("\u00ab", "").replace("\u00bb", "")
    return hashlib.sha256(re.sub(r"\s+", " ", s).encode("utf-8")).hexdigest()


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower().translate(TRANSLIT))
    return s.strip("_") or hashlib.md5(name.encode()).hexdigest()[:8]


# ------------------------------------------------------------- чанкинг
def is_heading(line):
    return bool(HEADING_MD_RE.match(line) or HEADING_NUM_RE.match(line)
                or (HEADING_WORD_RE.match(line) and len(line) < 120))


def chunk_fixed(text, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """Вариант А: фиксированный размер с перекрытием, границы по абзацам."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(buf.strip())
            buf = (buf[-overlap:] + "\n" + p).strip()
        else:
            buf = (buf + "\n" + p).strip()
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def split_sections(text):
    """[(заголовок, [абзацы]), ...]"""
    sections, current, header = [], [], None
    for line in text.split("\n"):
        if is_heading(line):
            if current:
                sections.append((header, current))
            header, current = line.strip().lstrip("# ").strip(), []
        else:
            current.append(line)
    if current:
        sections.append((header, current))
    return sections


def chunk_semantic(text, max_chars=CHUNK_MAX_CHARS, min_chars=CHUNK_MIN_CHARS):
    """Вариант Б: по смысловым блокам. Мелкие абзацы склеиваются до min_chars,
    блоки крупнее окна режутся с сохранением заголовка-шапки."""
    chunks = []
    for header, paras in split_sections(text):
        paras = [p.strip() for p in paras if p.strip()]
        if not paras:
            continue
        header_line = (header + "\n") if header else ""
        buf = ""
        for p in paras:
            if len(p) > max_chars:
                if buf:
                    chunks.append((header_line + buf).strip())
                    buf = ""
                for i in range(0, len(p), max_chars):
                    part = p[max(0, i - 50):i + max_chars] if i else p[i:i + max_chars]
                    chunks.append((header_line + part).strip())
                continue
            if len(buf) + len(p) + 1 > max_chars and len(buf) >= min_chars:
                chunks.append((header_line + buf).strip())
                buf = p
            else:
                buf = (buf + "\n" + p).strip()
        if buf:
            chunks.append((header_line + buf).strip())

    # склейка мелких чанков (в том числе между секциями), пока влезает в окно
    merged = []
    for ch in chunks:
        if merged and len(ch) < min_chars and len(merged[-1]) + len(ch) + 1 <= max_chars:
            merged[-1] = merged[-1] + "\n" + ch
        elif merged and len(merged[-1]) < min_chars and len(merged[-1]) + len(ch) + 1 <= max_chars:
            merged[-1] = merged[-1] + "\n" + ch
        else:
            merged.append(ch)
    return merged


# ------------------------------------------------------------- служебное
def load_index(path):
    idx = {}
    if Path(path).exists():
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                idx[row["file"]] = row
    return idx


def page_offsets(cleaned_pages):
    offs, pos = [], 0
    for ptext in cleaned_pages:
        offs.append(pos)
        pos += len(ptext) + 2
    return offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/documents/docs_index.csv")
    ap.add_argument("--raw-dir", default="data/documents/raw")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--max-chars", type=int, default=CHUNK_MAX_CHARS)
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    clean_dir = out_dir / "documents" / "clean"
    chunks_dir = out_dir / "chunks"
    clean_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # стартуем с чистых выходных файлов — база всегда пересобирается с нуля
    for name in ("chunks_fixed.jsonl", "chunks_semantic.jsonl"):
        (chunks_dir / name).unlink(missing_ok=True)

    index = load_index(args.index)
    seen, duplicates, n_docs = {}, [], 0

    files = sorted(p for p in raw_dir.iterdir()
                   if p.suffix.lower() in (".pdf", ".docx", ".doc", ".txt", ".md")
                   and not p.name.startswith("~"))
    if not files:
        print(f"[!] В {raw_dir} нет файлов. Скачайте документы и заполните {args.index}.")
        return

    for path in files:
        meta = index.get(path.name, {})
        title = meta.get("document_title") or path.stem
        print(f"[i] {path.name} -> {title}")
        pages = extract(path)
        cleaned_pages = [clean_text(t) for t in pages]
        full_text = "\n\n".join(cleaned_pages).strip()

        if not full_text:
            print("    [!] Пустой текст (вероятно сканированный PDF — нужен OCR). Пропущен.")
            continue

        h = normalize_for_hash(full_text)
        if h in seen:
            duplicates.append({"file": path.name, "duplicate_of": seen[h]})
            print(f"    [=] дубликат {seen[h]} — пропущен")
            continue
        seen[h] = path.name
        n_docs += 1

        doc_id = slugify(path.stem)
        md = [f"# {title}", "", f"Источник: {meta.get('source_url', '—')}  ",
              f"Дата: {meta.get('date', '—')}  ", "", full_text]
        (clean_dir / f"{doc_id}.md").write_text("\n".join(md), encoding="utf-8")

        offs = page_offsets(cleaned_pages)

        def make_chunks(chunks, variant):
            out = []
            for i, ch in enumerate(chunks, 1):
                first_line = ch.split("\n", 1)[0]
                start = full_text.find(first_line[:60])
                page = bisect.bisect_right(offs, start if start >= 0 else 0) if len(pages) > 1 else 1
                out.append({
                    "document_id": doc_id,
                    "document_title": title,
                    "source": meta.get("source_url", "—"),
                    "date": meta.get("date", "—"),
                    "document_type": meta.get("document_type", "unknown"),
                    "department": meta.get("department", "general"),
                    "page": page,
                    "chunk_id": f"{doc_id}__{variant}__{i:03d}",
                    "variant": variant,
                    "section": first_line[:120] if is_heading(first_line) else "",
                    "text": ch,
                })
            return out

        fixed = make_chunks(chunk_fixed(full_text, args.max_chars), "fixed")
        semantic = make_chunks(chunk_semantic(full_text, args.max_chars), "semantic")
        for fname, data in (("chunks_fixed.jsonl", fixed), ("chunks_semantic.jsonl", semantic)):
            with open(chunks_dir / fname, "a", encoding="utf-8") as f:
                for c in data:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"    [ok] чанков: fixed={len(fixed)}, semantic={len(semantic)}")

    with open(out_dir / "documents" / "duplicates_report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "duplicate_of"])
        w.writeheader()
        w.writerows(duplicates)

    counts = {}
    for fname in ("chunks_fixed.jsonl", "chunks_semantic.jsonl"):
        p = chunks_dir / fname
        counts[fname] = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
    print(f"\n[done] документов: {n_docs}, дубликатов удалено: {len(duplicates)}, "
          f"чанков fixed: {counts['chunks_fixed.jsonl']}, semantic: {counts['chunks_semantic.jsonl']}")


if __name__ == "__main__":
    main()
