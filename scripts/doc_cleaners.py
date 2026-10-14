"""Turn documentation sources into plain markdown-like text.

Each file is cleaned by its extension (markdown/MDX, reStructuredText, nginx XML, man pages).
The source's `format` adds a dialect step that resolves site-specific markup first:
Hugo shortcodes (Kubernetes, Docker), Liquid (GitHub docs) and JSX/MDX imports (OpenTofu).
Includes and code samples are inlined, because the examples are the most useful part.
"""

import html
import re
import subprocess
import tomllib
from functools import cache
from pathlib import Path

import yaml

FENCE = re.compile(r"^(```|~~~)", re.M)


# ---------------------------------------------------------------- shared helpers

def split_front_matter(text):
    """Return (front matter dict, body) for '---' delimited YAML front matter."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            try:
                meta = yaml.safe_load(text[3:end]) or {}
            except yaml.YAMLError:
                meta = {}
            return (meta if isinstance(meta, dict) else {}), text[end + 4:]
    return {}, text


def outside_fences(text, fn):
    """Apply fn only to the parts of text that are not inside fenced code blocks."""
    parts = re.split(r"(^(?:```|~~~)[^\n]*\n.*?^(?:```|~~~)[ \t]*$)", text, flags=re.M | re.S)
    return "".join(part if i % 2 else fn(part) for i, part in enumerate(parts))


def humanize_path(url):
    slug = url.split("#")[0].split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return slug.replace("-", " ").replace("_", " ") or url


def strip_markdown_noise(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)                       # images
    text = re.sub(r"\[AUTOTITLE\]\(([^)]*)\)", lambda m: humanize_path(m.group(1)), text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)                    # inline links -> text
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)                   # reference links
    text = re.sub(r"^\s*\[[^\]]+\]:\s*\S+.*$", "", text, flags=re.M)        # link definitions
    text = re.sub(r"\{:[^}]*\}", "", text)                                  # kramdown attributes
    return html_to_text(text)


HTML_TAGS = ("a|abbr|b|blockquote|br|caption|code|col|colgroup|dd|del|details|div|dl|dt|em|figcaption|figure|h[1-6]|hr|"
             "i|iframe|img|ins|kbd|li|mark|ol|p|picture|pre|q|s|samp|small|source|span|strong|sub|summary|sup|table|"
             "tbody|td|tfoot|th|thead|tr|u|ul|var|video|wbr")


def html_inline(text):
    # Kubernetes API reference cells: <code>field</code>&nbsp;<strong>*</strong><em>Type</em>
    text = re.sub(r"<code>(.*?)</code>(?:&nbsp;|\s)*(<strong>\*</strong>)?\s*<em>(.*?)</em>",
                  lambda m: f"`{m.group(1)}` ({m.group(3)}{', required' if m.group(2) else ''})", text, flags=re.S)
    text = re.sub(r"<(?:code|kbd|samp|var)\b[^>]*>(.*?)</(?:code|kbd|samp|var)>", r"`\1`", text, flags=re.S)
    text = re.sub(r"<(?:strong|b)\b[^>]*>(.*?)</(?:strong|b)>", r"**\1**", text, flags=re.S)
    text = re.sub(r"<(?:em|i)\b[^>]*>(.*?)</(?:em|i)>", r"*\1*", text, flags=re.S)
    text = re.sub(r"<br\s*/?>", "\n", text)
    return text


def html_to_text(text):
    """Convert HTML fragments embedded in markdown. Only known HTML tags are removed, so placeholders like <pod-name> survive."""
    if not re.search(rf"</?(?:{HTML_TAGS})\b", text):
        return text
    text = re.sub(r"<pre\b[^>]*>(.*?)</pre>",
                  lambda m: "\n```\n" + html.unescape(re.sub(rf"</?(?:{HTML_TAGS})\b[^>]*>", "", m.group(1))).strip("\n") + "\n```\n",
                  text, flags=re.S)

    def table(m):
        out = []
        caption = re.search(r"<caption\b[^>]*>(.*?)</caption>", m.group(0), re.S)
        if caption:
            out.append(caption.group(1).strip() + ":")
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", m.group(0), re.S):
            cells = re.findall(r"<t([hd])\b[^>]*>(.*?)</t[hd]>", row, re.S)
            if not cells or all(kind == "h" for kind, _ in cells) and not out[1:]:
                continue  # header row
            cells = [" ".join(html_inline(c).split()) for _, c in cells]
            out.append("- " + cells[0] + (": " + " | ".join(cells[1:]) if len(cells) > 1 else ""))
        return "\n" + "\n".join(out) + "\n"

    text = re.sub(r"<table\b.*?</table>", table, text, flags=re.S)
    text = html_inline(text)
    text = re.sub(r"<h([1-6])\b[^>]*>(.*?)</h\1>", lambda m: "\n" + "#" * int(m.group(1)) + " " + m.group(2).strip() + "\n", text, flags=re.S)
    text = re.sub(r"<(?:li|dt)\b[^>]*>", "\n- ", text)
    text = re.sub(r"<dd\b[^>]*>", "\n  ", text)
    text = re.sub(r"</?(?:p|div|ul|ol|dl|blockquote|details|summary)\b[^>]*>", "\n", text)
    text = re.sub(rf"</?(?:{HTML_TAGS})\b[^>]*>", "", text)
    return html.unescape(text)


def finish(text):
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def code_lang(path):
    return {".yaml": "yaml", ".yml": "yaml", ".json": "json", ".tf": "hcl", ".hcl": "hcl",
            ".sh": "shell", ".py": "python", ".go": "go", ".conf": "nginx"}.get(Path(path).suffix, "")


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- Hugo (Kubernetes, Docker)

K8S_HEADINGS = {
    "prerequisites": "Before you begin", "objectives": "Objectives", "whatsnext": "What's next",
    "cleanup": "Cleaning up", "synopsis": "Synopsis", "options": "Options", "seealso": "See also",
    "lessoncontent": "Lesson content", "options_from_parent": "Options inherited from parent commands",
}
SHORTCODE = r"\{\{[<%]\s*"
SHORTCODE_END = r"\s*[>%]\}\}"


@cache
def hugo_params(repo):
    repo = Path(repo)
    if (repo / "hugo.toml").exists():
        return tomllib.loads(read(repo / "hugo.toml")).get("params", {})
    if (repo / "hugo.yaml").exists():
        return (yaml.safe_load(read(repo / "hugo.yaml")) or {}).get("params", {})
    return {}


def shortcode_arg(args, name):
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', args)
    return m.group(1) if m else None


def hugo(text, repo, includes_dir, examples_dir=None, depth=0):
    params = hugo_params(str(repo))

    def include(m):
        target = Path(includes_dir) / m.group(1)
        if depth > 3 or not target.exists():
            return ""
        _, body = split_front_matter(read(target))
        return hugo(body, repo, includes_dir, examples_dir, depth + 1)

    def code_sample(m):
        file = shortcode_arg(m.group(1), "file")
        if not file or not examples_dir or not (Path(examples_dir) / file).exists():
            return ""
        return f"\n```{code_lang(file)}\n{read(Path(examples_dir) / file).rstrip()}\n```\n"

    def param(m):
        value = params.get(m.group(1), "")
        return str(value) if not isinstance(value, (dict, list)) else ""

    def skew(m):
        latest = str(params.get("latest", params.get("version", ""))).lstrip("v")
        if not re.fullmatch(r"\d+\.\d+", latest):
            return ""
        major, minor = map(int, latest.split("."))
        args = m.group(1).split()
        offsets = {"currentVersion": 0, "latestVersion": 0, "prevMinorVersion": -1, "nextMinorVersion": 1}
        if args and args[0] in offsets:
            return f"{major}.{minor + offsets[args[0]]}"
        if len(args) == 2 and args[0] == "currentVersionAddMinor":
            return f"{major}.{minor + int(args[1])}"
        return latest

    def feature_state(m):
        version, state = shortcode_arg(m.group(1), "for_k8s_version"), shortcode_arg(m.group(1), "state")
        return f"FEATURE STATE: Kubernetes {version} [{state}]" if version and state else ""

    def tab(m):
        name = shortcode_arg(m.group(1), "name")
        return f"\n**{name}:**\n" if name else ""

    rules = [
        (rf"{SHORTCODE}(comment|mermaid)\b.*?{SHORTCODE_END}.*?{SHORTCODE}/\1{SHORTCODE_END}", "", re.S),
        (rf'{SHORTCODE}include\s+"([^"]+)"{SHORTCODE_END}', include, 0),
        (rf"{SHORTCODE}(?:code_sample|codenew)\b(.*?){SHORTCODE_END}", code_sample, 0),
        (rf'{SHORTCODE}param\s+"([^"]+)"{SHORTCODE_END}', param, 0),
        (rf"{SHORTCODE}skew\s+(.*?){SHORTCODE_END}", skew, 0),
        (rf"{SHORTCODE}feature-state\b(.*?){SHORTCODE_END}", feature_state, 0),
        (rf'{SHORTCODE}heading\s+"([^"]+)"{SHORTCODE_END}', lambda m: "## " + K8S_HEADINGS.get(m.group(1), m.group(1)), 0),
        (rf'{SHORTCODE}glossary_tooltip\b(.*?){SHORTCODE_END}', lambda m: shortcode_arg(m.group(1), "text") or "", 0),
        (rf"{SHORTCODE}(note|caution|warning|important|tip)\s*{SHORTCODE_END}", lambda m: f"{m.group(1).capitalize()}:", 0),
        (rf"{SHORTCODE}tab\b(.*?){SHORTCODE_END}", tab, 0),
        (rf"{SHORTCODE}/?[^}}]*?{SHORTCODE_END}", "", 0),                      # anything else: drop the tag, keep inner text
    ]
    for pattern, repl, flags in rules:
        text = re.sub(pattern, repl, text, flags=flags)
    return text


# ---------------------------------------------------------------- Liquid (GitHub docs)

@cache
def github_variables(repo):
    variables = {}
    for file in (Path(repo) / "data" / "variables").glob("*.yml"):
        variables[file.stem] = yaml.safe_load(read(file)) or {}
    return variables


def liquid_condition(cond):
    """Evaluate a GitHub docs `ifversion` condition for github.com (fpt). Unknown feature names count as available."""
    def term(t):
        t = t.strip()
        if t.startswith("not "):
            return not term(t[4:])
        name = t.split()[0] if t else ""
        if name == "fpt":
            return True
        if name in ("ghes", "ghec", "ghae"):
            return False
        return True
    return any(all(term(t) for t in part.split(" and ")) for part in cond.split(" or "))


def liquid_versions(text):
    out, stack = [], []  # stack entries: [parent_active, branch_taken, current_active]
    pos = 0
    for m in re.finditer(r"\{%-?\s*(ifversion|elsif|else|endif)\b\s*(.*?)\s*-?%\}", text):
        active = all(frame[2] for frame in stack)
        if active:
            out.append(text[pos:m.start()])
        tag, cond = m.group(1), m.group(2)
        if tag == "ifversion":
            parent = all(frame[2] for frame in stack)
            ok = liquid_condition(cond)
            stack.append([parent, ok, ok])
        elif stack and tag == "elsif":
            frame = stack[-1]
            ok = not frame[1] and liquid_condition(cond)
            frame[2], frame[1] = ok, frame[1] or ok
        elif stack and tag == "else":
            frame = stack[-1]
            frame[2], frame[1] = not frame[1], True
        elif stack and tag == "endif":
            stack.pop()
        pos = m.end()
    if all(frame[2] for frame in stack):
        out.append(text[pos:])
    return "".join(out)


def liquid(text, repo, depth=0):
    variables = github_variables(str(repo))
    text = liquid_versions(text)

    def data(m):
        ref = m.group(1).split(".")
        if ref[0] == "variables" and len(ref) >= 3:
            value = variables.get(ref[1], {})
            for key in ref[2:]:
                value = value.get(key, "") if isinstance(value, dict) else ""
            return liquid(str(value), repo, depth + 1) if depth < 4 else str(value)
        if ref[0] == "reusables":
            target = Path(repo) / "data" / "reusables" / Path(*ref[1:-1]) / f"{ref[-1]}.md"
            return liquid(read(target), repo, depth + 1) if target.exists() and depth < 4 else ""
        return ""

    text = re.sub(r"\{%-?\s*data\s+([\w.-]+)\s*-?%\}", data, text)
    text = re.sub(r"\{%-?\s*(?:raw|endraw)\s*-?%\}", "", text)
    text = re.sub(r"\{%-?\s*(note|warning|tip|caution|important)\s*-?%\}", lambda m: f"{m.group(1).capitalize()}:", text)
    text = re.sub(r"\{%-?.*?-?%\}", "", text, flags=re.S)                    # octicon, tool switchers, etc.
    return text


# ---------------------------------------------------------------- MDX (OpenTofu, Docusaurus)

def mdx(text, path):
    imports = {}
    for m in re.finditer(r"^import\s+(\w+)\s+from\s+['\"]!!raw-loader!([^'\"]+)['\"];?\s*$", text, re.M):
        target = (Path(path).parent / m.group(2)).resolve()
        if target.exists():
            imports[m.group(1)] = (read(target).rstrip(), code_lang(target))
    text = re.sub(r"^(?:import|export)\s.*$", "", text, flags=re.M)

    def codeblock(m):
        attrs, name = m.group(1), m.group(2)
        if name not in imports:
            return ""
        body, lang = imports[name]
        lang = shortcode_arg(attrs, "language") or lang
        return f"\n```{lang}\n{body}\n```\n"

    text = re.sub(r"<CodeBlock\b([^>]*)>\s*\{(\w+)\}\s*</CodeBlock>", codeblock, text)
    text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.S)
    text = re.sub(r'<TabItem\b[^>]*label=\{?"([^"]+)"\}?[^>]*>', r"\n**\1:**\n", text)
    text = re.sub(r"^:::(\w+)(?:\s+(.*))?$", lambda m: f"{(m.group(2) or m.group(1)).capitalize()}:", text, flags=re.M)
    text = re.sub(r"^:::\s*$", "", text, flags=re.M)
    text = re.sub(r"</?[A-Z]\w*\b[^>]*/?>", "", text)                       # remaining JSX components
    text = re.sub(r"</?code>", "`", text)
    text = text.replace("\\*", "*")
    return text


# ---------------------------------------------------------------- reStructuredText (unbound)

def rst(text):
    lines, out, i = text.split("\n"), [], 0
    levels = []
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # heading: text line followed by an underline of punctuation at least as long
        if line.strip() and re.fullmatch(r"([=\-~^\"'`#*+])\1+", nxt.strip()) and len(nxt.strip()) >= len(line.strip()):
            char = nxt.strip()[0]
            if char not in levels:
                levels.append(char)
            out.append("#" * min(levels.index(char) + 1, 6) + " " + line.strip())
            i += 2
            continue
        # directives with an indented body
        m = re.match(r"^(\s*)\.\.\s+([\w:-]+)::\s*(.*)$", line)
        if m or line.rstrip().endswith("::"):
            indent = len(m.group(1)) if m else len(line) - len(line.lstrip())
            directive = m.group(2) if m else "literal"
            body, i = [], i + 1
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
                body.append(lines[i])
                i += 1
            while body and not body[-1].strip():
                body.pop()
            while body and not body[0].strip():
                body.pop(0)
            options = [b for b in body if re.match(r"^\s*:[\w-]+:", b)]
            body = [b for b in body if b not in options]
            dedent = min((len(b) - len(b.lstrip()) for b in body if b.strip()), default=0)
            body = [b[dedent:] for b in body]
            if directive in ("code-block", "code", "sourcecode", "literal"):
                lead = line.rstrip()[:-2].rstrip() if not m else ""
                if lead:
                    out.append(lead + ":")
                out += ["```" + (m.group(3) if m else ""), *body, "```"]
            elif directive in ("note", "warning", "tip", "important", "caution", "attention", "seealso"):
                out += [f"{directive.capitalize()}: " + (m.group(3) or ""), rst("\n".join(body))]
            elif directive in ("describe", "option", "confval", "envvar"):
                out += [f"**{m.group(3)}**", *body]
            # toctree, image, figure, index, include, only, raw, ...: dropped
            continue
        # comments, link targets and substitution definitions, with their indented continuation lines
        if re.match(r"^\s*\.\.(\s|$)", line):
            indent = len(line) - len(line.lstrip())
            i += 1
            while i < len(lines) and lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) > indent:
                i += 1
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    text = re.sub(r":[\w:]+:`([^`<]*?)\s*<[^>]*>`", r"\1", text)            # :ref:`text <target>`
    text = re.sub(r":[\w:]+:`([^`]*)`", r"`\1`", text)                       # :option:`x`, :term:`x`
    text = re.sub(r"`([^`<]+?)\s*<[^>]+>`_+", r"\1", text)                    # external links
    text = re.sub(r"\|[\w -]+\|", "", text)                                   # substitutions
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = re.sub(r"@@UAHL@[^@]+@([^@]+)@@", r"\1", text)                      # unbound man page anchors
    text = text.replace("@ub_conf_file@", "/etc/unbound/unbound.conf")         # build-time path substitution
    return text


# ---------------------------------------------------------------- nginx.org XML

def nginx_xml(text):
    text = re.sub(r"<\?xml.*?\?>|<!DOCTYPE.*?>|<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<example>(.*?)</example>", lambda m: "\n```nginx\n" + m.group(1).strip("\n") + "\n```\n", text, flags=re.S)
    text = re.sub(r'<(?:module|article)\b[^>]*name="([^"]+)"[^>]*>', r"# \1\n", text)
    text = re.sub(r'<section\b[^>]*name="([^"]+)"[^>]*>', r"\n## \1\n", text)

    def directive(m):
        # syntax lines in the XML omit the directive name; the rendered site prepends it
        name, body = m.group(1), m.group(2)
        block = lambda attrs: " { ... }" if 'block="yes"' in attrs else ";"  # noqa: E731
        body = re.sub(r"<syntax\b([^>/]*)>(.*?)</syntax>",
                      lambda s: f"<syntax>{name} {s.group(2)}{block(s.group(1))}</syntax>", body, flags=re.S)
        body = re.sub(r"<syntax\b([^>]*)/>", lambda s: f"<syntax>{name}{block(s.group(1))}</syntax>", body)
        return f"\n### {name}\n{body}"

    text = re.sub(r'<directive\b[^>]*name="([^"]+)"[^>]*>(.*?)</directive>', directive, text, flags=re.S)

    def syntax(m):
        inner = re.sub(r"<[^>]+>", "", m.group(1))
        return "\nSyntax: `" + " ".join(inner.split()) + "`\n"

    text = re.sub(r"<syntax\b[^>]*>(.*?)</syntax>", syntax, text, flags=re.S)
    text = re.sub(r"<default/>", "\nDefault: none\n", text)
    text = re.sub(r"<default>(.*?)</default>", lambda m: "\nDefault: `" + " ".join(re.sub(r"<[^>]+>", "", m.group(1)).split()) + "`\n", text, flags=re.S)
    text = re.sub(r"<context>(.*?)</context>", r"\nContext: \1\n", text)
    text = re.sub(r"<appeared-in>(.*?)</appeared-in>", r"\nAppeared in version \1.\n", text)
    text = re.sub(r'<tag-name>(.*?)</tag-name>', r"\n- \1:", text, flags=re.S)
    text = re.sub(r"<listitem>", "\n- ", text)
    text = re.sub(r"<para>|</para>", "\n\n", text)
    text = re.sub(r"<(?:literal|command|value|var|path|header|http-status)\b[^>]*>(.*?)</\w+>", r"`\1`", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text.replace("&nbsp;", " "))
    # XML line-wraps prose; join wrapped lines outside code blocks
    text = outside_fences(text, lambda t: re.sub(r"(?<=[^\n])\n(?=[^\n#\-`])", " ", t))
    text = re.sub(r"\nContext: (.*)\n\s*\nContext: ", r"\nContext: \1, ", text)
    while re.search(r"Context: [^\n]*\n\s*\nContext: ", text):
        text = re.sub(r"Context: ([^\n]*)\n\s*\nContext: ", r"Context: \1, ", text)
    return text


# ---------------------------------------------------------------- man pages (OpenRC)

def man(path):
    for cmd in (["mandoc", "-T", "utf8", str(path)], ["groff", "-man", "-T", "utf8", "-P", "-c", str(path)]):
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        out = re.sub(r".\x08", "", out)                                      # overstrike bold/underline
        return f"# {Path(path).name} man page\n\n{out}"
    raise RuntimeError("neither mandoc nor groff is installed, cannot render man pages")


# ---------------------------------------------------------------- entry point

def clean(src, repo, path):
    """Return (title, cleaned text) for one documentation file."""
    path, repo = Path(path), Path(repo)
    if re.fullmatch(r"\.[1-8]", path.suffix):
        return path.name, finish(man(path))
    text = read(path)
    if path.suffix == ".xml":
        text = nginx_xml(text)
        m = re.search(r"^# (.+)$", text, re.M)
        return (m.group(1) if m else path.stem), finish(text)
    meta, text = split_front_matter(text)
    title = str(meta.get("title") or meta.get("linkTitle") or meta.get("sidebar_label") or "")
    fmt = src.get("format", "markdown")
    if fmt == "hugo-k8s":
        text = hugo(text, repo, repo / "content/en/includes", repo / "content/en/examples")
    elif fmt == "hugo-docker":
        text = hugo(text, repo, repo / "content/includes")
    elif fmt == "liquid-github":
        title = liquid(title, repo)
        text = liquid(text, repo)
    elif path.suffix == ".mdx":
        text = mdx(text, path)
    if path.suffix == ".rst":
        text = rst(text)
    text = outside_fences(text, strip_markdown_noise)
    if not title:
        m = re.search(r"^#\s+(.+)$", text, re.M)
        title = m.group(1).strip() if m else path.stem
    elif not re.match(r"\s*#\s", text):
        text = f"# {title}\n\n{text}"
    return title, finish(text)
