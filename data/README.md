# База знаний ИИ-ассистента МГТУ им. Г.И. Носова

Репозиторий этапа подготовки данных для RAG-ассистента.
Конечный результат — воспроизводимая база знаний: запустив скрипты по инструкции,
можно получить ровно ту же базу заново.

## Структура

```
data/
├── README.md                 # этот файл: что здесь лежит и как всё воспроизвести
├── sources.csv               # перечень источников (что, откуда, приоритет, статус)
├── documents/
│   ├── raw/                  # (не в git) сырые скачанные файлы: pdf, docx, html, txt
│   ├── docs_index.csv        # реестр документов: файл -> название, источник, дата, тип, отдел
│   ├── duplicates_report.csv # отчёт дедупликации: какой документ дубликат чего
│   └── clean/                # очищенные документы (markdown), создаёт prepare_data.py
└── chunks/
    ├── chunks_fixed.jsonl    # вариант А: фиксированный размер + перекрытие
    └── chunks_semantic.jsonl # вариант Б: по смысловым блокам (заголовки/абзацы/пункты)
```

Отчёт об этапа: `docs/data_preparation.md`.

## Как воспроизвести

1. Скачать сырые документы в `data/documents/raw/`
   (имена файлов — латиницей, без пробелов; соответствие файл ↔ источник
   описывается в `data/documents/docs_index.csv`).
2. Установить зависимости: `pip install pymupdf python-docx scikit-learn`
3. Запустить подготовку:

   ```bash
   python scripts/prepare_data.py \
       --index data/documents/docs_index.csv \
       --raw-dir data/documents/raw \
       --out-dir data
   ```

   Скрипт сам: извлечёт текст, почистит, удалит дубликаты, создаст
   `data/documents/clean/`, `duplicates_report.csv` и два варианта чанков.

4. Сравнить варианты нарезки:

   ```bash
   python scripts/evaluate.py \
       --chunks data/chunks/chunks_fixed.jsonl data/chunks/chunks_semantic.jsonl \
       --questions scripts/questions.csv
   ```

   Скрипт считает recall@5 на тестовых вопросах (`scripts/questions.csv`):
   доля вопросов, для которых нужный фрагмент попал в топ-5 поиска
   (TF-IDF). Итог сравнения фиксируется в `docs/data_preparation.md`.

## Метаданные чанка

Каждая строка JSONL — один фрагмент:

| Поле | Смысл |
|---|---|
| `document_id` | идентификатор документа (slug имени файла) |
| `document_title` | название документа |
| `source` | URL источника |
| `date` | дата документа / последней редакции |
| `document_type` | тип: admission_rules, charter, policy, page и т.п. |
| `department` | тематика: admissions, dormitory, education и т.п. |
| `page` | страница в оригинале (для PDF; для сайтов — «—») |
| `chunk_id` | `<document_id>__<variant>__<n>` |
| `text` | текст фрагмента |

Дополнительно (наше решение, может расширяться): `section` — хлебные крошки
заголовков («Раздел 5 > Приложение 2»), `variant` — способ нарезки.

## Правила обновления базы знаний (кратко)

Полный текст — в `docs/data_preparation.md`, раздел «Правила обновления».
