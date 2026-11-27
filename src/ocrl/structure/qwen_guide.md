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

## Decisions taken for now

- A paragraph that opens with a dash, a bullet, or a number or letter with its mark is a `list_item`: every line of it, not only the first. Several such paragraphs in a row are several list items.
- A numbered heading is a `heading`.
- A label such as "ملخص" or "Abstract" standing above an abstract belongs to the `abstract`.
- Text on a magazine page: the short introduction between the title and the body is `standfirst`; large display text repeating a sentence of the body is `pull_quote`.
- Text printed inside a photograph, drawing or chart is `figure`.
- Marks that are not text (specks, stamps, ornaments, show-through read as words) are `noise`; writing added by hand is `handwriting`.
- Arabic reads right to left: on a page with columns the rightmost column comes first.
- A paragraph or list item that runs on into the next column or the next page is still one block.
