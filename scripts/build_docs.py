import os
import re
import shutil
from pathlib import Path
import markdown

DOCS_DIR = Path("docs")
OUT_DIR = Path("docs_dist")

CSS = """
:root {
    --bg-color: #0f172a;
    --text-color: #f8fafc;
    --accent: #3b82f6;
    --accent-hover: #60a5fa;
    --nav-bg: rgba(15, 23, 42, 0.75);
    --card-bg: rgba(30, 41, 59, 0.5);
    --border: rgba(255, 255, 255, 0.08);
}
body {
    margin: 0;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background-color: var(--bg-color);
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.15), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.15), transparent 25%);
    background-attachment: fixed;
    color: var(--text-color);
    line-height: 1.7;
    display: flex;
    min-height: 100vh;
}
aside {
    width: 300px;
    background: var(--nav-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-right: 1px solid var(--border);
    padding: 2.5rem 2rem;
    position: fixed;
    height: 100vh;
    overflow-y: auto;
    box-sizing: border-box;
    z-index: 10;
}
aside h2 {
    font-size: 1.5rem;
    font-weight: 800;
    margin-top: 0;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.025em;
}
aside ul {
    list-style: none;
    padding: 0;
    margin: 0;
}
aside li {
    margin-bottom: 0.5rem;
}
aside .nav-group {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}
aside a {
    color: #cbd5e1;
    text-decoration: none;
    font-size: 0.95rem;
    font-weight: 500;
    transition: all 0.2s ease;
    display: block;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    margin-left: -0.75rem;
}
aside a:hover {
    color: #fff;
    background: rgba(255, 255, 255, 0.05);
    transform: translateX(4px);
}
main {
    flex: 1;
    margin-left: 300px;
    padding: 4rem 10%;
    max-width: 900px;
    box-sizing: border-box;
    animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
h1, h2, h3, h4 {
    color: #f1f5f9;
    font-weight: 700;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    letter-spacing: -0.025em;
}
h1 { font-size: 3rem; margin-top: 0; border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 2rem; }
h2 { font-size: 2rem; }
h3 { font-size: 1.5rem; }
a { color: var(--accent); text-decoration: none; transition: color 0.2s; }
a:hover { color: var(--accent-hover); text-decoration: underline; }
p, li { color: #cbd5e1; }
code {
    background: rgba(15, 23, 42, 0.8);
    padding: 0.2rem 0.4rem;
    border-radius: 6px;
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 0.85em;
    color: #93c5fd;
    border: 1px solid var(--border);
}
pre {
    background: #0f172a;
    border: 1px solid var(--border);
    padding: 1.5rem;
    border-radius: 12px;
    overflow-x: auto;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    margin: 1.5rem 0;
}
pre code {
    background: transparent;
    padding: 0;
    color: #e2e8f0;
    border: none;
}
table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; margin-bottom: 1.5rem; }
th, td { padding: 1rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--card-bg); font-weight: 600; color: #f8fafc; }
tr:hover td { background: rgba(255,255,255,0.02); }
blockquote {
    border-left: 4px solid var(--accent);
    margin: 1.5rem 0;
    padding: 1rem 1.5rem;
    background: var(--card-bg);
    border-radius: 0 12px 12px 0;
    font-style: italic;
    color: #94a3b8;
}
img { max-width: 100%; border-radius: 12px; border: 1px solid var(--border); }
hr { border: 0; border-top: 1px solid var(--border); margin: 3rem 0; }
@media (max-width: 768px) {
    body { flex-direction: column; }
    aside { width: 100%; position: static; height: auto; padding: 2rem 1.5rem; border-right: none; border-bottom: 1px solid var(--border); }
    main { margin-left: 0; padding: 2rem 1.5rem; }
    h1 { font-size: 2.25rem; }
}
/* Pygments syntax highlighting base styles */
.codehilite .hll { background-color: #2b3b4e }
.codehilite .c { color: #6272a4 } /* Comment */
.codehilite .k { color: #ff79c6 } /* Keyword */
.codehilite .n { color: #f8f8f2 } /* Name */
.codehilite .s { color: #f1fa8c } /* String */
.codehilite .p { color: #f8f8f2 } /* Punctuation */
.codehilite .o { color: #ff79c6 } /* Operator */
.codehilite .nf { color: #50fa7b } /* Name.Function */
.codehilite .nc { color: #8be9fd } /* Name.Class */
"""

JS = """
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if(target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});
"""

def generate_nav(out_dir_depth):
    prefix = "../" * out_dir_depth
    
    # Collect files
    items = []
    for p in DOCS_DIR.rglob("*.md"):
        rel_path = p.relative_to(DOCS_DIR)
        html_path = rel_path.with_suffix(".html").as_posix()
        name = p.stem.replace("-", " ").replace("_", " ").title()
        if p.stem == "index":
            name = "Home" if p.parent == DOCS_DIR else p.parent.name.title() + " Home"
        items.append((str(rel_path.parent), name, html_path))
    
    # Group by directory
    groups = {}
    for dirname, name, html_path in items:
        group_name = "Overview" if dirname == "." else dirname.replace("-", " ").replace("_", " ").title()
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append((name, html_path))
    
    # Build HTML
    nav_html = "<h2>ML Env Doctor</h2>"
    
    # Ensure "Overview" is first
    group_keys = list(groups.keys())
    if "Overview" in group_keys:
        group_keys.remove("Overview")
        group_keys = ["Overview"] + sorted(group_keys)
    else:
        group_keys = sorted(group_keys)

    for group in group_keys:
        nav_html += f"<div class='nav-group'>{group}</div><ul>"
        for name, html_path in sorted(groups[group], key=lambda x: x[0]):
            nav_html += f"<li><a href='{prefix}{html_path}'>{name}</a></li>"
        nav_html += "</ul>"
        
    return nav_html

def build_docs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()
    
    (OUT_DIR / "styles.css").write_text(CSS, encoding="utf-8")
    (OUT_DIR / "script.js").write_text(JS, encoding="utf-8")
    (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")
    
    for p in DOCS_DIR.rglob("*.md"):
        rel_path = p.relative_to(DOCS_DIR)
        out_path = OUT_DIR / rel_path.with_suffix(".html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = p.read_text(encoding="utf-8")
        
        # Replace .md links with .html links
        content = re.sub(r'\]\(([^)]+)\.md([^)]*)\)', r'](\1.html\2)', content)
        
        md = markdown.Markdown(extensions=['fenced_code', 'codehilite', 'tables', 'toc'])
        html_content = md.convert(content)
        
        depth = len(rel_path.parts) - 1
        prefix = "../" * depth
        nav = generate_nav(depth)
        
        title = p.stem.replace("-", " ").replace("_", " ").title()
        if p.stem == "index":
            title = "ML Environment Doctor"
            
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{prefix}styles.css">
</head>
<body>
    <aside>
        {nav}
    </aside>
    <main>
        {html_content}
    </main>
    <script src="{prefix}script.js"></script>
</body>
</html>"""
        
        out_path.write_text(full_html, encoding="utf-8")
        print(f"Built {out_path}")

if __name__ == "__main__":
    build_docs()
