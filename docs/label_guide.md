# Label guide (draft, 2026-11-27)

Every line Azure found on a page gets one of these types, and lines are grouped into blocks. This draft was written by Claude from the owner's fixed type list;
the owner corrects it, and the corrected guide is what any labelling model is given.

| Type | What it is |
|---|---|
| `paragraph` | Running text. |
| `heading` | A heading inside the text, at any level. |
| `list_item` | One item of a list, all its lines: it opens with a dash, a bullet, a number or a letter mark. |
| `table` | Every line inside a table: cells and header rows. |
| `table_caption` | The title or caption of a table. |
| `reference` | One entry of a reference list or bibliography, all its lines. |
| `footnote` | A note at the foot of the page or the end of the text, with its mark. |
| `running_head` | Text repeated at the top of pages: journal, issue, article title. |
| `running_foot` | Text repeated at the foot of pages. |
| `page_number` | The printed page number. |
| `noise` | Not text: scan marks, stamps, show-through, specks that were read as words. |
| `title` | The title of the article, thesis or book, where it is set as the title. |
| `section_label` | A short label above the title naming the section or series. |
| `author` | The line naming who wrote it. |
| `supervisor` | The line naming a thesis supervisor. |
| `affiliation` | Where the author or supervisor works: post, department, university. |
| `degree_statement` | The statement of the degree a thesis was submitted for. |
| `standfirst` | The short introduction set between a magazine title and its text. |
| `abstract` | The summary of the article, in any language. |
| `keywords` | The keywords line. |
| `figure` | Text inside a picture, chart or diagram. |
| `figure_caption` | The caption of a picture, chart or diagram. |
| `pull_quote` | Display text that repeats a point from the running text. |
| `equation` | A formula set on its own line. |
| `verse` | Lines of poetry. |
| `handwriting` | Writing added by hand. |
| `other` | Anything that fits none of the above. |

## Blocks

- A block is one paragraph, one list item, one reference, one footnote, one heading, or one whole table.
- A block that runs over a column or page break is still one block.
- Every line belongs to exactly one block.

## Open questions for the owner

1. A paragraph that opens with a dash, such as a question in a list of research questions: list item (the hand keys say yes) or paragraph?
2. A numbered heading (`١- مقدمة`): heading, not list item?
3. The word `ملخص` or `Abstract` on its own line above an abstract: heading, or part of the abstract?
4. Notes printed directly under a table: table, footnote, or paragraph?
5. A table of contents: list items, a table, or other?
