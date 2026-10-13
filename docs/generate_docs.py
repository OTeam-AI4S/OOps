from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig

ROOT = Path(__file__).resolve().parents[1]


def on_pre_build(config: MkDocsConfig) -> None:
    for source, target in (
        ("README.md", "index.md"),
        ("README.zh.md", "index.zh.md"),
        ("license.md", "about/license.md"),
        ("license.zh.md", "about/license.zh.md"),
        ("license-faq.md", "about/license-faq.md"),
        ("license-faq.zh.md", "about/license-faq.zh.md"),
        ("privacy.md", "about/privacy.md"),
        ("privacy.zh.md", "about/privacy.zh.md"),
    ):
        destination = Path(config.docs_dir) / target
        content = (ROOT / source).read_text(encoding="utf-8")
        if source.startswith("README"):
            for page in ("license", "license.zh", "license-faq", "license-faq.zh", "privacy", "privacy.zh"):
                content = content.replace(f"({page}.md)", f"(about/{page}.md)")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.read_text(encoding="utf-8") != content:
            destination.write_text(content, encoding="utf-8")
