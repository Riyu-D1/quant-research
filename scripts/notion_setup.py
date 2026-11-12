"""One-time Notion setup for the paper bot.

    NOTION_TOKEN=ntn_xxx NOTION_PARENT_PAGE_ID=<page> \
        .venv/bin/python scripts/notion_setup.py

Creates the "Trade Journal" database under the parent page (the Notion
integration must already have edit access to that page — share the page
with the integration first, see docs/BOT_SETUP.md) and prints the new
database id. Save that id as the NOTION_DB_ID GitHub secret; afterwards
the bot doesn't need NOTION_PARENT_PAGE_ID.

If NOTION_DB_ID is already set, this just verifies the token can read it.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from quant_research.notion_journal import NotionJournal


def main() -> None:
    token = os.environ.get("NOTION_TOKEN", "")
    db = os.environ.get("NOTION_DB_ID", "")
    parent = os.environ.get("NOTION_PARENT_PAGE_ID", "")
    if not token:
        sys.exit("NOTION_TOKEN missing — create an internal integration at "
                 "https://www.notion.so/profile/integrations (see "
                 "docs/BOT_SETUP.md)")
    if db:
        j = NotionJournal(token, db_id=db)
        d = j._req("GET", f"/databases/{db}")
        title = "".join(t["plain_text"] for t in d.get("title", []))
        print(f"NOTION_DB_ID {db} reachable — title: {title!r}")
        return
    if not parent:
        sys.exit("neither NOTION_DB_ID nor NOTION_PARENT_PAGE_ID set")
    j = NotionJournal(token, parent_page_id=parent)
    print(f"created database id: {j.db}")
    print("save it as the NOTION_DB_ID GitHub secret — "
          "gh secret set NOTION_DB_ID")


if __name__ == "__main__":
    main()
