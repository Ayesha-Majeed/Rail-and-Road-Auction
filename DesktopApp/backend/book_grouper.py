import re
import os
from datetime import datetime
from collections import defaultdict

class BookGrouper:
    # Supports underscores, hyphens, and spaces: BookID_001.jpg, BookID-001.jpg, BookID 001.jpg
    PATTERN = re.compile(r"^(.+?)[_\-\s](\d+)\.(jpg|jpeg|png|tiff|bmp|webp)$", re.IGNORECASE)

    def group(self, folder: str, recursive: bool = False) -> dict:
        books = defaultdict(list)
        if recursive:
            for root, _, files in os.walk(folder):
                for fn in files:
                    if fn.startswith("."): continue
                    m = self.PATTERN.match(fn)
                    if m:
                        book_id  = m.group(1).strip()
                        if not book_id: continue
                        page_num = int(m.group(2))
                        books[book_id].append((page_num, os.path.join(root, fn)))
        else:
            for fn in os.listdir(folder):
                if fn.startswith("."): continue
                fp = os.path.join(folder, fn)
                if not os.path.isfile(fp):
                    continue
                m = self.PATTERN.match(fn)
                if m:
                    book_id  = m.group(1).strip()
                    if not book_id: continue
                    page_num = int(m.group(2))
                    books[book_id].append((page_num, fp))
        for book_id in books:
            books[book_id].sort(key=lambda x: x[0])
        return dict(books)

    def build_document(self, book_id: str, pages: list) -> dict:
        total = len(pages)
        doc = {
            "book_id":        book_id,
            "total_pages":    total,
            "front_cover":    None,
            "back_cover":     None,
            "interior_pages": [],
            "status":         "complete",
            "synced_at":      datetime.now().isoformat(),
        }
        page_nums      = [p[0] for p in pages]
        has_duplicates = len(page_nums) != len(set(page_nums))
        missing        = self._check_missing(page_nums)

        if has_duplicates:
            doc["status"] = "warning_duplicates"
        if missing:
            doc["status"] = "warning_missing_pages"
            doc["missing_pages"] = missing

        sorted_nums = sorted(page_nums)
        for page_num, filepath in pages:
            fname = os.path.basename(filepath)
            entry = {"page_id": f"{book_id}_{str(page_num).zfill(3)}",
                     "file_name": fname, "file_path": filepath}
            if total == 2:
                if page_num == min(page_nums):
                    entry["note"] = "Combined Front & Back Cover"
                    doc["front_cover"] = entry
                else:
                    entry.update({"page_number": page_num, "type": "interior"})
                    doc["interior_pages"].append(entry)
            else:
                if page_num == sorted_nums[0]:
                    doc["front_cover"] = entry
                elif page_num == sorted_nums[1]:
                    doc["back_cover"] = entry
                else:
                    entry.update({"page_number": page_num, "type": "interior"})
                    doc["interior_pages"].append(entry)
        return doc

    def _check_missing(self, page_nums):
        if not page_nums:
            return []
        full = set(range(min(page_nums), max(page_nums) + 1))
        return sorted(full - set(page_nums))
