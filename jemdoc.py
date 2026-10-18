#!/usr/bin/env python3

"""new_jemdoc: the project's jemdoc 0.7.3 parser with a simplified HTML renderer.

Usage: python3 new_jemdoc.py -o new_index.html index.jemdoc
Uses only the Python standard library. Text, sections, menus and publications
come from the .jemdoc, MENU and .bib sources. The generated HTML embeds its
stylesheet, citation controls, bibliography downloads and local images.
The legacy jemdoc.css requested by existing sources is replaced by the new
theme. Other custom styles/scripts and ordinary jemdoc blocks are retained.
"""

# Copyright (C) 2007-2012 Jacob Mattingley (jacobm@stanford.edu).
#
# This file is part of jemdoc.
#
# jemdoc is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# jemdoc is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#
# The LaTeX equation portions of this file were initially based on
# latexmath2png, by Kamil Kisiel (kamil@kamikisiel.net).
#

import sys
import os
import re
import time
import io
from subprocess import *
import tempfile

def info():
  print(__doc__)
  print('Platform: ' + sys.platform + '.')
  print('Python: %s, located at %s.' % (sys.version[:5], sys.executable))
  print('Equation support:', end=' ')
  (supported, message) = testeqsupport()
  if supported:
    print('yes.')
  else:
    print('no.')
  print(message)

def testeqsupport():
  supported = True
  msg = ''
  p = Popen('latex --version', shell=True, stdout=PIPE, stderr=PIPE)
  rc = p.wait()
  if rc != 0:
    msg += '  latex: not found.\n'
    supported = False
  else:
    msg += '  latex: ' + p.stdout.readlines()[0].rstrip() + '.\n'
  p = Popen('dvipng --version', shell=True, stdout=PIPE, stderr=PIPE)
  rc = p.wait()
  if rc != 0:
    msg += '  dvipng: not found.\n'
    supported = False
  else:
    msg += '  dvipng: ' + p.stdout.readlines()[0].rstrip() + '.\n'

  return (supported, msg[:-1])

class controlstruct(object):
  def __init__(self, infile, outfile=None, conf=None, inname=None, eqs=True,
         eqdir='eqs', eqdpi=130):
    self.inname = inname
    self.inf = infile
    self.outf = outfile
    self.conf = conf
    self.linenum = 0
    self.otherfiles = []
    self.eqs = eqs
    self.eqdir = eqdir
    self.eqdpi = eqdpi
    self.eqsupport = True
    self.eqcache = True
    self.eqpackages = []
    self.texlines = []
    self.analytics = None
    self.eqbd = {} 
    self.baseline = None

  def pushfile(self, newfile):
    self.otherfiles.insert(0, self.inf)
    self.inf = open(resolve_source_path(self, newfile), 'rb')

  def pushfile_string(self, text_content):
    """Allows feeding a generated string into the reader pipeline."""
    self.otherfiles.insert(0, self.inf)
    source_name = getattr(self.inf, 'name', self.inname)
    self.inf = io.BytesIO(bytes(text_content, encoding='utf-8'))
    self.inf.name = source_name

  def nextfile(self):
    self.inf.close()
    self.inf = self.otherfiles.pop(0)

def showhelp():
  a = """Usage: python3 new_jemdoc.py [OPTIONS] SOURCEFILE [SOURCEFILE ...] 
  Produces html markup from a jemdoc SOURCEFILE.

  Most of the time you can use jemdoc without any additional flags.
  For example, typing

    python3 new_jemdoc.py -o new_index.html index.jemdoc

  will produce a standalone new_index.html from index.jemdoc, using the new
  configuration.

  Some configuration options can be overridden by specifying a
  configuration file.  You can use

    python3 new_jemdoc.py --show-config

  to print a sample configuration file (which includes all of the
  default options). Any or all of the configuration [blocks] can be
  overwritten by including them in a configuration file, and running,
  for example,

    python3 new_jemdoc.py -c mywebsite.conf index.jemdoc 

  You can view version and installation details with

    python3 new_jemdoc.py --version

  See https://github.com/mengzili/jemdoc-python3 for many more details."""
  b = ''
  for l in a.splitlines(True):
    if l.startswith(' '*4):
      b += l[4:]
    else:
      b += l

  print(b)

def standardconf():
  a = """[firstbit]
  <!doctype html>
  <html lang="en">
  <head>
  <meta charset="utf-8" />
  <meta name="generator" content="new_jemdoc (based on jemdoc)" />
  
  [defaultcss]
  <link rel="stylesheet" href="jemdoc.css" type="text/css" />
  
  [windowtitle]
  # used in header for window title.
  <title>|</title>

  [fwtitlestart]
  <div id="fwtitle">

  [fwtitleend]
  </div>
  
  [doctitle]
  # used at top of document.
  <div id="toptitle">
  <h1>|</h1>
  
  [subtitle]
  <div id="subtitle">|</div>
  
  [doctitleend]
  </div>
  
  [bodystart]
  </head>
  <body>
  
  [analytics]
  <script type="text/javascript">
  var gaJsHost = (("https:" == document.location.protocol) ? "https://ssl." : "http://www.");
  document.write(unescape("%3Cscript src='" + gaJsHost + "google-analytics.com/ga.js' type='text/javascript'%3E%3C/script%3E"));
  </script>
  <script type="text/javascript">
  try {
      var pageTracker = _gat._getTracker("|");
      pageTracker._trackPageview();
  } catch(err) {}</script>
  
  [menustart]
  <div id="tlayout">
  <div id="layout-menu">
  
  [menuend]
  </div>
  <div id="layout-content">
  
  [menucategory]
  <div class="menu-category">|</div>

  [menuitem]
  <div class="menu-item"><a href="|1">|2</a></div>

  [specificcss]
  <link rel="stylesheet" href="|" type="text/css" />

  [specificjs]
  <script src="|.js" type="text/javascript"></script>
  
  [currentmenuitem]
  <div class="menu-item"><a href="|1" class="current">|2</a></div>
  
  [nomenu]
  <div id="layout-content">
  
  [menulastbit]
  </div>
  </div>
  
  [nomenulastbit]
  </div>
  
  [bodyend]
  </body>
  </html>
  
  [infoblock]
  <div class="infoblock">
  
  [codeblock]
  <div class="codeblock">
  
  [blocktitle]
  <div class="blocktitle">|</div>
  
  [infoblockcontent]
  <div class="blockcontent">
  
  [codeblockcontent]
  <div class="blockcontent"><pre>
  
  [codeblockend]
  </pre></div></div>
  
  [codeblockcontenttt]
  <div class="blockcontent"><tt class="tthl">
  
  [codeblockendtt]
  </tt></div></div>
  
  [infoblockend]
  </div></div>
  
  [footerstart]
  <div id="footer">
  <div id="footer-text">
  
  [footerend]
  </div>
  </div>
  
  [lastupdated]
  Page generated |, by a slightly modified version of <a href="https://github.com/jem/jemdoc">jemdoc</a>.

  [sourcelink]
  (<a href="|" target="_blank">source</a>)

  """
  b = ''
  for l in a.splitlines(True):
    if l.startswith('  '):
      b += l[2:]
    else:
      b += l

  return b

class JandalError(Exception):
  pass

class NoEqSupport(Exception):
  pass

def format_bib_categorized(filename, f_control):
    """Parse a .bib file and render categorized publication lists for jemdoc.

    BibTeX grouping braces are used for parsing/protection only and are not
    emitted in the visible text. Common LaTeX accents are converted to Unicode.
    HAL, arXiv and DOI identifiers are rendered as links when available.
    """
    import re
    import unicodedata
    import base64

    def latex_to_unicode(value):
        """Convert common BibTeX/LaTeX text constructs to displayable Unicode."""
        if not value:
            return ''

        value = value.replace('\n', ' ').replace('\r', ' ')

        combining = {
            '"': '\u0308',  # diaeresis
            "'": '\u0301',  # acute
            '`': '\u0300',  # grave
            '^': '\u0302',  # circumflex
            '~': '\u0303',  # tilde
            '=': '\u0304',  # macron
            '.': '\u0307',  # dot above
            'u': '\u0306',  # breve
            'v': '\u030c',  # caron
            'H': '\u030b',  # double acute
            'c': '\u0327',  # cedilla
            'k': '\u0328',  # ogonek
            'r': '\u030a',  # ring above
        }

        def accent_repl(match):
            accent = match.group(1)
            letter = match.group(2)
            return unicodedata.normalize('NFC', letter + combining[accent])

        # Standard forms: \\"i, \\"{i}, {\\"i}, {\\"{i}}, \\c{c}, ...
        accent_pattern = re.compile(
            r'\\(["\'`\^~=\.uvHckr])\s*\{?\\?([A-Za-z])\}?'
        )
        previous = None
        while previous != value:
            previous = value
            value = accent_pattern.sub(accent_repl, value)

        # Also accept the frequently encountered malformed/legacy form {"i}
        # where the backslash before the accent marker is absent.
        malformed_accent_pattern = re.compile(r'\{(["\'`\^~])\s*([A-Za-z])\}')
        value = malformed_accent_pattern.sub(accent_repl, value)

        # Common LaTeX letter commands used in names and titles.
        letter_commands = {
            r'\\ae': 'æ', r'\\AE': 'Æ',
            r'\\oe': 'œ', r'\\OE': 'Œ',
            r'\\aa': 'å', r'\\AA': 'Å',
            r'\\o': 'ø', r'\\O': 'Ø',
            r'\\l': 'ł', r'\\L': 'Ł',
            r'\\ss': 'ß',
            r'\\i': 'ı', r'\\j': 'ȷ',
        }
        for command, replacement in letter_commands.items():
            value = re.sub(command + r'(?=\s|[{}]|$)', replacement, value)

        # TeX escapes that should be displayed literally.
        value = re.sub(r'\\([&%#$._])', r'\1', value)
        value = value.replace('~', ' ')

        # Remaining braces are BibTeX grouping/protection braces, not content.
        value = value.replace('{', '').replace('}', '')

        # Collapse whitespace introduced by formatting/group removal.
        value = re.sub(r'\s+', ' ', value).strip()
        return value

    def camel_case_title(value):
        """Capitalize the first letter of each title word without
        lowercasing the rest, so acronyms such as VLSF, MIMO or 6G are
        preserved exactly as written in the BibTeX source.
        """
        if not value:
            return ''

        # Capitalize at the beginning of the title and after separators that
        # naturally start a new word. Do not treat apostrophes as separators
        # (e.g. "User's Guide", not "User'S Guide").
        pattern = re.compile(r'(^|[\s\-–—/:;\(\[])'
                             r'([^\W\d_])', re.UNICODE)
        return pattern.sub(
            lambda m: m.group(1) + m.group(2).upper(),
            value,
        )

    def normalize_arxiv_id(value):
        value = latex_to_unicode(value).strip()
        value = re.sub(r'^arxiv:\s*', '', value, flags=re.IGNORECASE)
        m = re.search(
            r'(\d{4}\.\d{4,5}(?:v\d+)?|[A-Za-z.-]+/\d{7}(?:v\d+)?)',
            value,
            re.IGNORECASE,
        )
        return m.group(1) if m else ''

    def normalize_doi(value):
        value = latex_to_unicode(value).strip()
        value = re.sub(r'^doi:\s*', '', value, flags=re.IGNORECASE)
        value = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', value, flags=re.IGNORECASE)
        return value.strip()

    def normalize_hal_id(value):
        value = latex_to_unicode(value).strip()
        if not value:
            return ''
        value = re.sub(r'^https?://[^/]+/', '', value, flags=re.IGNORECASE)
        value = value.strip('/')
        # HAL URLs can contain extra path components; the identifier is first.
        if '/' in value:
            value = value.split('/', 1)[0]
        return value

    try:
        with open(filename, 'r', encoding='utf-8') as bibf:
            content = bibf.read()
    except IOError as exc:
        raise JandalError("Could not open bibliography %s: %s" % (filename, exc))

    records = []

    # Locate all occurrences of @type. Entry bodies are then extracted with
    # balanced braces so nested BibTeX groups are handled correctly.
    entry_starts = list(re.finditer(r'@([a-zA-Z]+)', content))

    for start_match in entry_starts:
        entry_type = start_match.group(1).strip().lower()

        search_start = start_match.end()
        brace_start = content.find('{', search_start)
        if brace_start == -1:
            continue

        brace_count = 0
        body = ''
        raw_bibtex = ''
        for idx in range(brace_start, len(content)):
            char = content[idx]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            if brace_count == 0:
                body = content[brace_start + 1:idx]
                # Keep the original BibTeX entry verbatim for the citation popup.
                raw_bibtex = content[start_match.start():idx + 1].strip()
                break

        if not body:
            continue

        # Parse key/value pairs. Quoted values may contain protected groups and
        # escaped quote accents (e.g. \\"i), so only an unescaped quote at brace
        # depth zero terminates a quoted field.
        fields = {}
        i = 0
        first_comma = body.find(',')
        if first_comma != -1:
            i = first_comma + 1

        while i < len(body):
            while i < len(body) and (body[i].isspace() or body[i] == ','):
                i += 1
            if i >= len(body):
                break

            equal_sign = body.find('=', i)
            if equal_sign == -1:
                break

            key_name = body[i:equal_sign].strip().lower()
            i = equal_sign + 1

            while i < len(body) and body[i].isspace():
                i += 1
            if i >= len(body):
                break

            val_chars = []
            if body[i] == '{':
                value_brace_depth = 1
                i += 1
                while i < len(body):
                    char = body[i]
                    if char == '{':
                        value_brace_depth += 1
                    elif char == '}':
                        value_brace_depth -= 1

                    if value_brace_depth == 0:
                        i += 1
                        break

                    val_chars.append(char)
                    i += 1

            elif body[i] == '"':
                i += 1
                quoted_brace_depth = 0
                while i < len(body):
                    char = body[i]
                    if char == '{':
                        quoted_brace_depth += 1
                    elif char == '}' and quoted_brace_depth > 0:
                        quoted_brace_depth -= 1

                    # Count preceding backslashes to distinguish an escaped
                    # quote from the quote that ends the BibTeX field.
                    if char == '"' and quoted_brace_depth == 0:
                        backslashes = 0
                        j = i - 1
                        while j >= 0 and body[j] == '\\':
                            backslashes += 1
                            j -= 1
                        if backslashes % 2 == 0:
                            i += 1
                            break

                    val_chars.append(char)
                    i += 1

            else:
                while i < len(body) and body[i] != ',' and not body[i].isspace():
                    val_chars.append(body[i])
                    i += 1

            fields[key_name] = latex_to_unicode(''.join(val_chars))

        # Field assembly mapping.
        author_parts = [
            part.strip() for part in fields.get('author', '').split(' and ') if part.strip()
        ]
        if len(author_parts) > 1:
            author = ', '.join(author_parts[:-1]) + ' and ' + author_parts[-1]
        elif author_parts:
            author = author_parts[0]
        else:
            author = ''
        title = fields.get('title', '')  # Preserve the title's BibTeX capitalization.
        venue = fields.get(
            'journal',
            fields.get(
                'booktitle',
                fields.get(
                    'school',
                    fields.get(
                        'institution',
                        fields.get('publisher', fields.get('howpublished', '')),
                    ),
                ),
            ),
        )
        year = fields.get('year', '')
        url = fields.get('url', fields.get('pdf', ''))
        doi = normalize_doi(fields.get('doi', ''))
        eprint = fields.get('eprint', '')
        eprint_type = fields.get('eprinttype', fields.get('archiveprefix', ''))
        note = fields.get('note', '')

        if not title and not author:
            continue

        sort_year = 0
        match_year = re.search(r'\d{4}', year)
        if match_year:
            sort_year = int(match_year.group())

        # HAL identifier: support common field names used by HAL/BibTeX exports,
        # then fall back to a HAL URL if present.  Keep a versioned identifier
        # for the target URL when available (e.g. hal-05754349v1), while the
        # visible label omits the version suffix (e.g. ⟨hal-05754349⟩).
        hal_id = ''
        hal_url_id = ''
        for hal_key in ('hal_id', 'halid', 'hal-id', 'hal'):
            if fields.get(hal_key):
                hal_id = normalize_hal_id(fields[hal_key])
                break

        hal_url = fields.get('hal_url', fields.get('halurl', ''))
        candidate_url = hal_url or (url if re.search(r'https?://(?:hal\.|[^/]*\.hal\.)|hal\.science|archives-ouvertes\.fr', url, re.IGNORECASE) else '')
        if candidate_url:
            m = re.search(
                r'https?://(?:[^/]+)/(?:[^/]+/)?((?:hal|tel|inria|cea|ensl|pasteur|cnrs)-[A-Za-z0-9._-]+)',
                candidate_url,
                re.IGNORECASE,
            )
            if m:
                hal_url_id = m.group(1)
                if not hal_id:
                    hal_id = hal_url_id

        if not hal_id and eprint and 'hal' in eprint_type.lower():
            hal_id = normalize_hal_id(eprint)

        hal_link_id = hal_url_id or hal_id
        hal_display_id = re.sub(r'v\d+$', '', hal_id or hal_link_id, flags=re.IGNORECASE)

        # arXiv identifier: accept explicit arXiv fields, eprint metadata, venue,
        # or arXiv URLs. Do not assume every generic eprint is arXiv.
        arxiv_id = ''
        for arxiv_key in ('arxiv', 'arxiv_id', 'arxivid'):
            if fields.get(arxiv_key):
                arxiv_id = normalize_arxiv_id(fields[arxiv_key])
                if arxiv_id:
                    break

        if not arxiv_id and eprint:
            looks_arxiv = (
                'arxiv' in eprint_type.lower()
                or bool(re.fullmatch(r'(?:arxiv:\s*)?(?:\d{4}\.\d{4,5}(?:v\d+)?|[A-Za-z.-]+/\d{7}(?:v\d+)?)', eprint, re.IGNORECASE))
            )
            if looks_arxiv:
                arxiv_id = normalize_arxiv_id(eprint)

        if not arxiv_id and venue and 'arxiv' in venue.lower():
            arxiv_id = normalize_arxiv_id(venue)
        if not arxiv_id and url and 'arxiv.org' in url.lower():
            arxiv_id = normalize_arxiv_id(url)

        if entry_type == 'article':
            category = 0
        elif entry_type in ('inproceeding', 'inproceedings', 'conference', 'proceedings'):
            category = 1
        elif entry_type in ('phdthesis', 'mastersthesis'):
            category = 3
            type_label = 'PhD thesis' if entry_type == 'phdthesis' else 'MSc thesis'
            venue = type_label + (' · ' + venue if venue else '')
        else:
            category = 2
            if not venue:
                venue = 'Preprint'

        links = []
        if hal_link_id:
            links.append(('HAL', 'https://hal.science/' + hal_link_id))
        if arxiv_id:
            links.append(('arXiv', 'https://arxiv.org/abs/' + arxiv_id))
        if doi:
            links.append(('DOI', 'https://doi.org/' + doi))
        if fields.get('pdf'):
            links.append(('PDF', fields['pdf']))
        if category == 3:
            nnt_source = fields.get('nnt', fields.get('nnt_id', fields.get('nntid', note)))
            nnt_match = re.search(r'(?:NNT\s*:?\s*)?([0-9]{4}[A-Za-z0-9._-]+)', nnt_source, re.IGNORECASE)
            if nnt_match:
                links.append(('Theses.fr', 'https://theses.fr/' + nnt_match.group(1)))
        if url and not links:
            links.append(('Read', url))
        records.append({'author': author, 'title': title, 'venue': venue,
                        'sort_year': sort_year, 'category': category,
                        'links': links, 'raw': raw_bibtex})

    # Feed a raw block through the existing include pipeline, bypassing prose
    # markup so BibTeX and publication punctuation remain intact. Newlines are
    # entities to prevent a standalone '~' in a citation ending the raw block.
    rendered = render_publications(records, filename, content).replace('\n', '&#10;')
    return '\n\n~\n{}{raw}\n' + rendered + '\n~\n\n'

def raisejandal(msg, line=0):
  if line == 0:
    s = "%s" % msg
  else:
    s = "line %d: %s" % (line, msg)
  raise JandalError(s)

def readnoncomment(f):
  l = f.readline().decode(encoding='utf-8')
  if l == '':
    return l
  elif l[0] == '#': # jem: be a little more generous with the comments we accept?
    return readnoncomment(f)
  else:
    return l.rstrip() + '\n' # leave just one \n and no spaces etc.

def parseconf(cns):
  syntax = {}
  warn = False # jem. make configurable?
  # manually add the defaults as a file handle.
  fs = [io.BytesIO(bytes(standardconf(), encoding='utf-8'))]
  for sname in cns:
    fs.append(open(sname, 'rb'))

  for f in fs:
    while pc(controlstruct(f)) != '':
      l = readnoncomment(f)
      r = re.match(r'\[(.*)\]\n', l)

      if r:
        tag = r.group(1)

        s = ''
        l = readnoncomment(f)
        while l not in ('\r', '\n', ''):
          s += l
          l = readnoncomment(f)

        syntax[tag] = s

    f.close()

  return syntax

def insertmenuitems(f, mname, current, prefix):
  m = open(resolve_source_path(f, mname), 'rb')
  while pc(controlstruct(m)) != '':
    l = readnoncomment(m)
    l = l.strip()
    if l == '':
      continue

    r = re.match(r'\s*(.*?)\s*\[(.*)\]', l)

    if r: # then we have a menu item.
      link = r.group(2)
      # Don't use prefix if we have an absolute link.
      if '://' not in r.group(2):
        link = prefix + allreplace(link)

      # replace spaces with nbsps.
      # do do this, even though css would make it work - ie ignores.
      # only replace spaces that aren't in {{ blocks.
      in_quote = False
      menuitem = ""
      for group in re.split(r'({{|}})', r.group(1)):
        if in_quote:
          if group == '}}':
            in_quote = False
            next
          else:
            menuitem += group
        else:
          if group == '{{':
            in_quote = True
            next
          else:
            menuitem += br(re.sub(r'(?<!\\n) +', '~', group), f)

      if link[-len(current):] == current:
        hb(f.outf, f.conf['currentmenuitem'], link, menuitem)
      else:
        hb(f.outf, f.conf['menuitem'], link, menuitem)

    else: # menu category.
      hb(f.outf, f.conf['menucategory'], br(l, f))

  m.close()

def out(f, s):
  f.write(s)

def hb(f, tag, content1, content2=None):
  """Writes out a halfblock (hb)."""

  if content1 is None:
    content1 = ""

  if content2 is None:
    out(f, re.sub(r'\|', content1, tag))
  else:
    r = re.sub(r'\|1', content1, tag)
    r = re.sub(r'\|2', content2, r)
    out(f, r)

def pc(f, ditchcomments=True):
  """Peeks at next character in the file."""
  # Should only be used to look at the first character of a new line.
  c = f.inf.read(1).decode(encoding='utf-8')
  if c: # only undo forward movement if we're not at the end.
    if ditchcomments and c == '#':
      l = nl(f)
      doincludes(f, l)
      return pc(f, ditchcomments)

    if c in ' \t':
      return pc(f)

    if c == '\\':
      c += pc(f)

    f.inf.seek(-1, 1)
  elif f.otherfiles:
    f.nextfile()
    return pc(f, ditchcomments)

  return c

def doincludes(f, l):
  ir = 'includeraw{'
  i = 'include{'
  ib = 'includebib{'
  
  if l.startswith(ir):
    with open(resolve_source_path(f, l[len(ir):-2]), encoding='utf-8') as nf:
      f.outf.write(embed_fragment_assets(nf.read(), Path(nf.name).parent))
    f.pushfile_string('\n')
  elif l.startswith(i):
    f.pushfile(l[len(i):-2])
  elif l.startswith(ib):
    # Process bib file, compile into dynamic jemdoc strings, and inject them into the open virtual file buffer
    bib_filename = resolve_source_path(f, l[len(ib):-2])
    generated_jemdoc = format_bib_categorized(bib_filename, f)
    
    # We turn the generated string into a byte stream so the jemdoc engine can process it mid-flight
    f.pushfile_string(generated_jemdoc)
  else:
    return False

  return True

def nl(f, withcount=False, codemode=False):
  """Get input file line."""
  f.block_source = getattr(f.inf, "name", f.inname)
  s = f.inf.readline().decode(encoding='utf-8')
  if not s and f.otherfiles:
    f.nextfile()
    return nl(f, withcount, codemode)

  f.linenum += 1

  if not codemode:
    # remove any special characters - assume they were checked by pc()
    # before we got here.
    # remove any trailing comments.
    s = s.lstrip(' \t')
    s = re.sub(r'\s*(?<!\\)#.*', '', s)

  if withcount:
    if s[0] == '.':
      m = r'\.'
    else:
      m = s[0]

    r = re.match('(%s+) ' % m, s)
    if not r:
      raise SyntaxError("couldn't handle the jandal (code 12039) on line"
                " %d" % f.linenum)

    if not codemode:
      s = s.lstrip('-.=:')

    return (s, len(r.group(1)))
  else:
    if not codemode:
      s = s.lstrip('-.=:')

    return s

def np(f, withcount=False, eatblanks=True):
  """Gets the next paragraph from the input file."""
  # New paragraph markers signalled by characters in following tuple.
  if withcount:
    (s, c) = nl(f, withcount)
  else:
    s = nl(f)

  block_source = getattr(f, 'block_source', f.inname)
  while pc(f) not in ('\r', '\n', '-', '.', ':', '', '=', '~', '{', '\\(', '\\)'):
    s += nl(f)

  while eatblanks and pc(f) in ('\n', '\r'):
    nl(f) # burn blank line.

  f.block_source = block_source

  # in both cases, ditch the trailing \n.
  if withcount:
    return (s[:-1], c)
  else:
    return s[:-1]

def quote(s):
  return re.sub(r"""[\\*/+"'<>&$%\.~[\]-]""", r'\\\g<0>', s)

def replacequoted(b):
  """Quotes {{raw html}} sections."""

  r = re.compile(r'\{\{(.*?)\}\}', re.M + re.S)
  m = r.search(b)
  while m:
    qb = quote(m.group(1))

    b = b[:m.start()] + qb + b[m.end():]

    m = r.search(b, m.start())

  return b

def replacepercents(b):
  # replace %sections% as +{{sections}}+. Do not replace if within a link.

  r = re.compile(r'(?<!\\)%(.*?)(?<!\\)%', re.M + re.S)
  m = r.search(b)
  while m:
    #qb = '+' + quote(m.group(1)) + '+'
    a = re.sub(r'\[', r'BSNOTLINKLEFT12039XX', m.group(1))
    a = re.sub(r'\]', r'BSNOTLINKRIGHT12039XX', a)
    qb = '+{{' + a + '}}+'

    b = b[:m.start()] + qb + b[m.end():]

    m = r.search(b, m.start())

  return b

def replaceequations(b, f):
  # replace $sections$ and \(sections\) as equations.
  rs = ((re.compile(r'(?<!\\)\$(.*?)(?<!\\)\$', re.M + re.S), False),
     (re.compile(r'(?<!\\)\\\((.*?)(?<!\\)\\\)', re.M + re.S), True))
  for (r, wl) in rs:
    m = r.search(b)
    while m:
      eq = m.group(1)
      if wl:
        fn = str(abs(hash(eq + 'wl120930alsdk')))
      else:
        fn = str(abs(hash(eq)))

      # Find out the baseline when we first encounter an equation (don't
      # bother, otherwise).
      # Other initialization stuff which we do only once we know we have
      # equations.
      if f.baseline is None:
        # See if the eqdir exists, and if not, create it.
        if not os.path.isdir(f.eqdir):
          os.mkdir(f.eqdir)

        # Check that the tools we need exist.
        (supported, message) = testeqsupport()
        if not supported:
          print('WARNING: equation support disabled.')
          print(message)
          f.eqsupport = False
          return b

        # Calculate the baseline.
        eqt = "0123456789xxxXXxX"
        (f.baseline, blfn) = geneq(f, eqt, dpi=f.eqdpi, wl=False,
                       outname='baseline-' + str(f.eqdpi))
        if os.path.exists(blfn):
          os.remove(blfn)

      fn = fn + '-' + str(f.eqdpi)
      (depth, fullfn) = geneq(f, eq, dpi=f.eqdpi, wl=wl, outname=fn)
      fullfn = fullfn.replace('\\', '/')

      offset = depth - f.baseline + 1

      eqtext = allreplace(eq)
      eqtext = eqtext.replace('\\', '')
      eqtext = eqtext.replace('\n', ' ')
      eqtext = eqtext.replace('\r', ' ')

      # Double braces will cause problems with escaping of image tag.
      eqtext = eqtext.replace('{{', 'DOUBLEOPENBRACE')
      eqtext = eqtext.replace('}}', 'DOUBLECLOSEBRACE')

      if wl:
        b = b[:m.start()] + \
            '{{\n<div class="eqwl"><img class="eqwl" src="%s" alt="%s" />\n<br /></div>}}' % (fullfn, eqtext) + b[m.end():]
      else:
        b = b[:m.start()] + \
          '{{<img class="eq" src="%s" alt="%s" style="vertical-align: -%dpx" />}}' % (fullfn, eqtext, offset) + b[m.end():]

      # jem: also clean out line breaks in the alttext?
      m = r.search(b, m.start())

  return replacequoted(b)

def replaceimages(b):
  # works with [img{width}{height}{alttext} location caption].
  r = re.compile(r'(?<!\\)\[img((?:\{.*?\}){,3})\s(.*?)(?:\s(.*?))?(?<!\\)\]',
           re.M + re.S)
  m = r.search(b)
  s = re.compile(r'{(.*?)}', re.M + re.S)
  while m:
    m1 = list(s.findall(m.group(1)))
    m1 += ['']*(3 - len(m1))

    bits = []
    link = m.group(2).strip()
    bits.append(r'src=\"%s\"' % quote(link))

    if m1[0]:
      if m1[0].isdigit():
        s = m1[0] + 'px'
      else:
        s = m1[0]
      bits.append(r'width=\"%s\"' % quote(s))
    if m1[1]:
      if m1[1].isdigit():
        s = m1[1] + 'px'
      else:
        s = m1[1]
      bits.append(r'height=\"%s\"' % quote(s))
    if m1[2]:
      bits.append(r'alt=\"%s\"' % quote(m1[2]))
    else:
      bits.append(r'alt=\"\"')

    b = b[:m.start()] + r'<img %s />' % " ".join(bits) + b[m.end():]

    m = r.search(b, m.start())

  return b

def replacelinks(b):
  # works with [link.html new link style].
  r = re.compile(r'(?<!\\)\[(.*?)(?:\s(.*?))?(?<!\\)\]', re.M + re.S)
  m = r.search(b)
  while m:
    m1 = m.group(1).strip()

    if '@' in m1 and not m1.startswith('mailto:') and not \
       m1.startswith(('http://', 'https://')):
      link = 'mailto:' + m1
    else:
      link = m1

    # first unquote any hashes (e.g. for in-page links).
    link = re.sub(r'\\#', '#', link)

    # remove any +{{ or }}+ links.
    link = re.sub(r'(\+\{\{|\}\}\+)', r'%', link)

    link = quote(link)

    if m.group(2):
      linkname = m.group(2).strip()
    else:
      # remove any mailto before labelling.
      linkname = re.sub('^mailto:', '', link)

    b = b[:m.start()] + r'<a href=\"%s\">%s<\/a>' % (link, linkname) + b[m.end():]

    m = r.search(b, m.start())

  return b

def br(b, f, tableblock=False):
  """Does simple text replacements on a block of text. ('block replacements')"""

  # Deal with environment variables (say, for Michael Grant).
  r = re.compile(r"!\$(\w{2,})\$!", re.M + re.S)

  for m in r.findall(b):
    repl = os.environ.get(m)
    if repl == None:
      b = re.sub(r"!\$%s\$!" % m, 'FAILED_MATCH_' + m, b)
    else:
      b = re.sub(r"!\$%s\$!" % m, repl, b)

  # Deal with literal backspaces.
  if f.eqs and f.eqsupport:
    b = replaceequations(b, f)

  b = re.sub(r'\\\\', 'jemLITerl33talBS', b)

  # Deal with {{html embedding}}.
  b = replacequoted(b)

  b = allreplace(b)

  b = b.lstrip('-. \t') # remove leading spaces, tabs, dashes, dots.
  b = replaceimages(b) # jem not sure if this is still used.

  # Slightly nasty hackery in this next bit.
  b = replacepercents(b)
  b = replacelinks(b)
  b = re.sub(r'BSNOTLINKLEFT12039XX', r'[', b)
  b = re.sub(r'BSNOTLINKRIGHT12039XX', r']', b)
  b = replacequoted(b)

  # Deal with /italics/ first because the '/' in other tags would otherwise
  # interfere.
  r = re.compile(r'(?<!\\)/(.*?)(?<!\\)/', re.M + re.S)
  b = re.sub(r, r'<i>\1</i>', b)

  # Deal with *bold*.
  r = re.compile(r'(?<!\\)\*(.*?)(?<!\\)\*', re.M + re.S)
  b = re.sub(r, r'<b>\1</b>', b)

  # Deal with +monospace+.
  r = re.compile(r'(?<!\\)\+(.*?)(?<!\\)\+', re.M + re.S)
  b = re.sub(r, r'<tt>\1</tt>', b)

  # Deal with "double quotes".
  r = re.compile(r'(?<!\\)"(.*?)(?<!\\)"', re.M + re.S)
  b = re.sub(r, r'&ldquo;\1&rdquo;', b)

  # Deal with left quote `.
  r = re.compile(r"(?<!\\)`", re.M + re.S)
  b = re.sub(r, r'&lsquo;', b)

  # Deal with apostrophe '.
  # Add an assertion that the next character's not a letter, to deal with
  # apostrophes properly.
  r = re.compile(r"(?<!\\)'(?![a-zA-Z])", re.M + re.S)
  b = re.sub(r, r'&rsquo;', b)

  # Deal with em dash ---.
  r = re.compile(r"(?<!\\)---", re.M + re.S)
  b = re.sub(r, r'&#8201;&mdash;&#8201;', b)

  # Deal with en dash --.
  r = re.compile(r"(?<!\\)--", re.M + re.S)
  b = re.sub(r, r'&ndash;', b)

  # Deal with ellipsis ....
  r = re.compile(r"(?<!\\)\.\.\.", re.M + re.S)
  b = re.sub(r, r'&hellip;', b)

  # Deal with non-breaking space ~.
  r = re.compile(r"(?<!\\)~", re.M + re.S)
  b = re.sub(r, r'&nbsp;', b)

  # Deal with registered trademark \R.
  r = re.compile(r"(?<!\\)\\R", re.M + re.S)
  b = re.sub(r, r'&reg;', b)

  # Deal with copyright \C.
  r = re.compile(r"(?<!\\)\\C", re.M + re.S)
  b = re.sub(r, r'&copy;', b)

  # Deal with middot \M.
  r = re.compile(r"(?<!\\)\\M", re.M + re.S)
  b = re.sub(r, r'&middot;', b)

  # Deal with line break.
  r = re.compile(r"(?<!\\)\\n", re.M + re.S)
  b = re.sub(r, r'<br />', b)

  # Deal with paragraph break. Caution! Should only use when we're already in
  # a paragraph.
  r = re.compile(r"(?<!\\)\\p", re.M + re.S)
  b = re.sub(r, r'</p><p>', b)

  if tableblock:
    # Deal with ||, meaning </td></tr><tr><td>
    r = re.compile(r"(?<!\\)\|\|", re.M + re.S)
    f.tablecol = 2
    bcopy = b
    b = ""
    r2 = re.compile(r"(?<!\\)\|", re.M + re.S)
    for l in bcopy.splitlines():
      f.tablerow += 1
      l = re.sub(r, r'</td></tr>\n<tr class="r%d"><td class="c1">' \
            % f.tablerow, l)

      l2 = ''
      col = 2
      r2s = r2.split(l)
      for x in r2s[:-1]:
        l2 += x + ('</td><td class="c%d">' % col)
        col += 1
      l2 += r2s[-1]

      b += l2

  # Second to last, remove any remaining quoting backslashes.
  b = re.sub(r'\\(?!\\)', '', b)

  # Deal with literal backspaces.
  b = re.sub('jemLITerl33talBS', r'\\', b)

  # Also fix up DOUBLEOPEN and DOUBLECLOSEBRACES.
  b = re.sub('DOUBLEOPENBRACE', '{{', b)
  b = re.sub('DOUBLECLOSEBRACE', '}}', b)

  return embed_fragment_assets(b, Path(getattr(f, "block_source", f.inname)).resolve().parent)

def allreplace(b):
  """Replacements that should be done on everything."""
  r = re.compile(r"(?<!\\)&", re.M + re.S)
  b = re.sub(r, r'&amp;', b)

  r = re.compile(r"(?<!\\)>", re.M + re.S)
  b = re.sub(r, r'&gt;', b)

  r = re.compile(r"(?<!\\)<", re.M + re.S)
  b = re.sub(r, r'&lt;', b)

  return b

def pyint(f, l):
  l = l.rstrip()
  l = allreplace(l)

  r = re.compile(r'(#.*)')
  l = r.sub(r'<span class = "comment">\1</span>', l)

  if l.startswith('&gt;&gt;&gt;'):
    hb(f, '<span class="pycommand">|</span>\n', l)
  else:
    out(f, l + '\n')

def putbsbs(l):
  for i in range(len(l)):
    l[i] = '\\b' + l[i] + '\\b'

  return l

def gethl(lang):
  # disable comments by default, by choosing unlikely regex.
  d = {'strings':False}
  if lang in ('py', 'python'):
    d['statement'] = ['break', 'continue', 'del', 'except', 'exec',
              'finally', 'pass', 'print', 'raise', 'return', 'try',
              'with', 'global', 'assert', 'lambda', 'yield', 'def',
              'class', 'for', 'while', 'if', 'elif', 'else',
              'import', 'from', 'as', 'assert']
    d['builtin'] = ['True', 'False', 'set', 'open', 'frozenset',
            'enumerate', 'object', 'hasattr', 'getattr', 'filter',
            'eval', 'zip', 'vars', 'unicode', 'type', 'str',
            'repr', 'round', 'range', 'and', 'in', 'is', 'not',
            'or']
    d['special'] = ['cols', 'optvar', 'param', 'problem', 'norm2', 'norm1',
            'value', 'minimize', 'maximize', 'rows', 'rand',
            'randn', 'printval', 'matrix']
    d['error'] = [r'\w*Error',]
    d['commentuntilend'] = '#'
    d['strings'] = True
  elif lang in ['c', 'c++', 'cpp']:
    d['statement'] = ['if', 'else', 'printf', 'return', 'for']
    d['builtin'] = ['static', 'typedef', 'int', 'float', 'double', 'void',
            'clock_t', 'struct', 'long', 'extern', 'char']
    d['operator'] = ['#include.*', '#define', '@pyval{', '}@', '@pyif{',
             '@py{']
    d['error'] = [r'\w*Error',]
    d['commentuntilend'] = ['//', '/*', ' * ', '*/']
  elif lang in ('rb', 'ruby'):
    d['statement'] = putbsbs(['while', 'until', 'unless', 'if', 'elsif',
                  'when', 'then', 'else', 'end', 'begin',
                  'rescue', 'class', 'def'])
    d['operator'] = putbsbs(['and', 'not', 'or'])
    d['builtin'] = putbsbs(['true', 'false', 'require', 'warn'])
    d['special'] = putbsbs(['IO'])
    d['error'] = putbsbs([r'\w*Error',])
    d['commentuntilend'] = '#'
    d['strings'] = True
    d['strings'] = True
    if lang in ['c++', 'cpp']:
      d['builtin'] += ['bool', 'virtual']
      d['statement'] += ['new', 'delete']
      d['operator'] += ['&lt;&lt;', '&gt;&gt;']
      d['special'] = ['public', 'private', 'protected', 'template',
              'ASSERT']
  elif lang == 'sh':
    d['statement'] = ['cd', 'ls', 'sudo', 'cat', 'alias', 'for', 'do',
              'done', 'in', ]
    d['operator'] = ['&gt;', r'\\', r'\|', ';', '2&gt;', 'monolith&gt;',
             'kiwi&gt;', 'ant&gt;', 'kakapo&gt;', 'client&gt;']
    d['builtin'] = putbsbs(['gem', 'gcc', 'python', 'curl', 'wget', 'ssh',
                'latex', 'find', 'sed', 'gs', 'grep', 'tee',
                'gzip', 'killall', 'echo', 'touch',
                'ifconfig', 'git', r'(?<!\.)tar(?!\.)'])
    d['commentuntilend'] = '#'
    d['strings'] = True
  elif lang == 'matlab':
    d['statement'] = putbsbs(['max', 'min', 'find', 'rand', 'cumsum', 'randn', 'help',
                     'error', 'if', 'end', 'for'])
    d['operator'] = ['&gt;', 'ans =', '>>', '~', r'\.\.\.']
    d['builtin'] = putbsbs(['csolve'])
    d['commentuntilend'] = '%'
    d['strings'] = True
  elif lang == 'commented':
    d['commentuntilend'] = '#'

  # Add bsbs (whatever those are).
  for x in ['statement', 'builtin', 'special', 'error']:
    if x in d:
      d[x] = putbsbs(d[x])

  return d

def language(f, l, hl):
  l = l.rstrip()
  l = allreplace(l)
  # handle strings.
  if hl['strings']:
    r = re.compile(r'(".*?")')
    l = r.sub(r'<span CLCLclass="string">\1</span>', l)
    r = re.compile(r"('.*?')")
    l = r.sub(r'<span CLCLclass="string">\1</span>', l)

  if 'statement' in hl:
    r = re.compile('(' + '|'.join(hl['statement']) + ')')
    l = r.sub(r'<span class="statement">\1</span>', l)

  if 'operator' in hl:
    r = re.compile('(' + '|'.join(hl['operator']) + ')')
    l = r.sub(r'<span class="operator">\1</span>', l)

  if 'builtin' in hl:
    r = re.compile('(' + '|'.join(hl['builtin']) + ')')
    l = r.sub(r'<span class="builtin">\1</span>', l)

  if 'special' in hl:
    r = re.compile('(' + '|'.join(hl['special']) + ')')
    l = r.sub(r'<span class="special">\1</span>', l)

  if 'error' in hl:
    r = re.compile('(' + '|'.join(hl['error']) + ')')
    l = r.sub(r'<span class="error">\1</span>', l)

  l = re.sub('CLCLclass', 'class', l)

  if 'commentuntilend' in hl:
    cue = hl['commentuntilend']
    if isinstance(cue, (list, tuple)):
      for x in cue:
        if l.strip().startswith(x):
          hb(f, '<span class="comment">|</span>\n', allreplace(l))
          return
        if '//' in cue: # Handle this separately.
          r = re.compile(r'\/\/.*')
          l = r.sub(r'<span class="comment">\g<0></span>', l)
    elif cue == '#': # Handle this separately.
      r = re.compile(r'#.*')
      l = r.sub(r'<span class="comment">\g<0></span>', l)
    elif cue == '%': # Handle this separately.
      r = re.compile(r'%.*')
      l = r.sub(r'<span class="comment">\g<0></span>', l)
    elif l.strip().startswith(cue):
      hb(f, '<span class="comment">|</span>\n', allreplace(l))
      return

  out(f, l + '\n')

def geneq(f, eq, dpi, wl, outname):
  # First check if there is an existing file.
  eqname = os.path.join(f.eqdir, outname + '.png')

  eqdepths = {}
  if f.eqcache:
    try:
      dc = open(os.path.join(f.eqdir, '.eqdepthcache'), 'rb')
      for l in dc:
        a = l.split()
        eqdepths[a[0]] = int(a[1])
      dc.close()

      if os.path.exists(eqname) and eqname in eqdepths:
        return (eqdepths[eqname], eqname)
    except IOError:
      print('eqdepthcache read failed.')

  # Open tex file.
  tempdir = tempfile.gettempdir()
  fd, texfile = tempfile.mkstemp('.tex', '', tempdir, True)
  basefile = texfile[:-4]
  g = os.fdopen(fd, 'wb')

  preamble = '\\documentclass{article}\n'
  for p in f.eqpackages:
    preamble += '\\usepackage{%s}\n' % p
  for p in f.texlines:
    # Replace \{ and \} in p with { and }.
    # XXX hack.
    preamble += re.sub(r'\\(?=[{}])', '', p + '\n')
  preamble += '\\pagestyle{empty}\n\\begin{document}\n'
  g.write(preamble)
  
  # Write the equation itself.
  if wl:
    g.write('\\[%s\\]' % eq)
  else:
    g.write('$%s$' % eq)

  # Finish off the tex file.
  g.write('\n\\newpage\n\\end{document}')
  g.close()

  exts = ['.tex', '.aux', '.dvi', '.log']
  try:
    # Generate the DVI file
    latexcmd = 'latex -file-line-error-style -interaction=nonstopmode ' + \
         '-output-directory %s %s' % (tempdir, texfile)
    p = Popen(latexcmd, shell=True, stdout=PIPE)
    rc = p.wait()
    if rc != 0:
      for l in p.stdout.readlines():
        print('  ' + l.rstrip())
      exts.remove('.tex')
      raise Exception('latex error')

    dvifile = basefile + '.dvi'
    dvicmd = 'dvipng --freetype0 -Q 9 -z 3 --depth -q -T tight -D %i -bg Transparent -o %s %s' % (dpi, eqname, dvifile)
    # discard warnings, as well.
    p = Popen(dvicmd, shell=True, stdout=PIPE, stderr=PIPE)
    rc = p.wait()
    if rc != 0:
      print(p.stderr.readlines())
      raise Exception('dvipng error')
    depth = int(p.stdout.readlines()[-1].split('=')[-1])
  finally:
    # Clean up.
    for ext in exts:
      g = basefile + ext
      if os.path.exists(g):
        os.remove(g)

  # Update the cache if we're using it.
  if f.eqcache and eqname not in eqdepths:
    try:
      dc = open(os.path.join(f.eqdir, '.eqdepthcache'), 'ab')
      dc.write(eqname + ' ' + str(depth) + '\n')
      dc.close()
    except IOError:
      print('eqdepthcache update failed.')
  return (depth, eqname)

def dashlist(f, ordered=False):
  level = 0

  if ordered:
    char = '.'
    ul = 'ol'
  else:
    char = '-'
    ul = 'ul'

  while pc(f) == char:
    (s, newlevel) = np(f, True, False)

    # first adjust list number as appropriate.
    if newlevel > level:
      for i in range(newlevel - level):
        if newlevel > 1:
          out(f.outf, '\n')
        out(f.outf, '<%s>\n<li>' % ul)
    elif newlevel < level:
      out(f.outf, '\n</li>')
      for i in range(level - newlevel):
        #out(f.outf, '</li>\n</%s>\n</li><li>' % ul)
        # demote means place '</ul></li>' in the file.
        out(f.outf, '</%s>\n</li>' % ul)
      #out(f.outf, '\n<li>')
      out(f.outf, '\n<li>')
    else:
      # same level, make a new list item.
      out(f.outf, '\n</li>\n<li>')

    out(f.outf, '<p>' + br(s, f) + '</p>')
    level = newlevel

  for i in range(level):
    out(f.outf, '\n</li>\n</%s>\n' % ul)

def colonlist(f):
  out(f.outf, '<dl>\n')
  while pc(f) == ':':
    s = np(f, eatblanks=False)
    r = re.compile(r'\s*{(.*?)(?<!\\)}(.*)', re.M + re.S)
    g = re.match(r, s)

    if not g or len(g.groups()) != 2:
      raise SyntaxError("couldn't handle the jandal (invalid deflist "
               "format) on line %d" % f.linenum)
    # split into definition / non-definition part.
    defpart = g.group(1)
    rest = g.group(2)

    hb(f.outf, '<dt>|</dt>\n', br(defpart, f))
    hb(f.outf, '<dd><p>|</p></dd>\n', br(rest, f))

  out(f.outf, '</dl>\n')

def codeblock(f, g):
  if g[1] == 'raw':
    raw = True
    ext_prog = None
  elif g[0] == 'filter_through':
    # Filter through external program.
    raw = False
    ext_prog = g[1]
    buff = ""
  else:
    ext_prog = None
    raw = False
    out(f.outf, f.conf['codeblock'])
    if g[0]:
      hb(f.outf, f.conf['blocktitle'], g[0])
    if g[1] == 'jemdoc':
      out(f.outf, f.conf['codeblockcontenttt'])
    else:
      out(f.outf, f.conf['codeblockcontent'])

  # Now we are handling code.
  # Handle \~ and ~ differently.
  stringmode = False
  while 1: # wait for EOF.
    l = nl(f, codemode=True)
    if not l:
      break
    elif l.startswith('~'):
      break
    elif l.startswith('\\~'):
      l = l[1:]
    elif l.startswith('\\{'):
      l = l[1:]
    elif ext_prog:
      buff += l
      continue
    elif stringmode:
      if l.rstrip().endswith('"""'):
        out(f.outf, l + '</span>')
        stringmode = False
      else:
        out(f.outf, l)
      continue

    # jem revise pyint out of the picture.
    if g[1] == 'pyint':
      pyint(f.outf, l)
    else:
      if raw:
        out(f.outf, embed_fragment_assets(l, Path(getattr(f, 'block_source', f.inname)).resolve().parent))
      elif g[1] == 'jemdoc':
        # doing this more nicely needs python 2.5.
        for x in ('#', '~', '>>>', r'\~', '{'):
          if str(l).lstrip().startswith(x):
            out(f.outf, '</tt><pre class="tthl">')
            out(f.outf, l + '</pre><tt class="tthl">')
            break
        else:
          for x in (':', '.', '-'):
            if str(l).lstrip().startswith(x):
              out(f.outf, '<br />' + prependnbsps(l))
              break
          else:
            if str(l).lstrip().startswith('='):
              out(f.outf, prependnbsps(l) + '<br />')
            else:
              out(f.outf, l)
      else:
        if l.startswith('\\#include{') or l.startswith('\\#includeraw{'):
          out(f.outf, l[1:])
        elif l.startswith('#') and doincludes(f, l[1:]):
          continue
        elif g[1] in ('python', 'py') and l.strip().startswith('"""'):
          out(f.outf, '<span class="string">' + l)
          stringmode = True
        else:
          language(f.outf, l, gethl(g[1]))

  if raw:
    return
  elif ext_prog:
    print('filtering through %s...' % ext_prog)

    output,_ = Popen(ext_prog, shell=True, stdin=PIPE,
                     stdout=PIPE).communicate(buff)
    out(f.outf, output)
  else:
    if g[1] == 'jemdoc':
      out(f.outf, f.conf['codeblockendtt'])
    else:
      out(f.outf, f.conf['codeblockend'])

def prependnbsps(l):
  g = re.search('(^ *)(.*)', l).groups()
  return g[0].replace(' ', '&nbsp;') + g[1]

def inserttitle(f, t):
  if t is not None:
    hb(f.outf, f.conf['doctitle'], t)

    # Look for a subtitle.
    if pc(f) not in ('\n', '\r'):
      hb(f.outf, f.conf['subtitle'], br(np(f), f))

    hb(f.outf, f.conf['doctitleend'], t)

def _procfile_jemdoc(f):
  f.linenum = 0

  menu = None
  # convert these to a dictionary.
  showfooter = True
  showsourcelink = False
  showlastupdated = True
  showlastupdatedtime = True
  nodefaultcss = False
  fwtitle = False
  css = []
  js = []
  title = None
  while pc(f, False) == '#':
    l = f.inf.readline().decode(encoding='utf-8')
    f.linenum += 1
    if doincludes(f, l[1:]):
      continue
    if l.startswith('# jemdoc:'):
      l = l[len('# jemdoc:'):]
      a = l.split(',')
      # jem only handle one argument for now.
      for b in a:
        b = b.strip()
        if b.startswith('menu'):
          sidemenu = True
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          g = re.findall(r, b)
          if len(g) > 3 or len(g) < 2:
            raise SyntaxError('sidemenu error on line %d' % f.linenum)

          if len(g) == 2:
            menu = (f, g[0], g[1], '')
          else:
            menu = (f, g[0], g[1], g[2])

        elif b.startswith('nofooter'):
          showfooter = False

        elif b.startswith('nodate'):
          showlastupdated = False

        elif b.startswith('notime'):
          showlastupdatedtime = False

        elif b.startswith('fwtitle'):
          fwtitle = True

        elif b.startswith('showsource'):
          showsourcelink = True

        elif b.startswith('nodefaultcss'):
          nodefaultcss = True

        elif b.startswith('addcss'):
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          css += re.findall(r, b)

        elif b.startswith('addjs'):
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          js += re.findall(r, b)

        elif b.startswith('addpackage'):
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          f.eqpackages += re.findall(r, b)

        elif b.startswith('addtex'):
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          f.texlines += re.findall(r, b)

        elif b.startswith('analytics'):
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          f.analytics = re.findall(r, b)[0]

        elif b.startswith('title'):
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          g = re.findall(r, b)
          if len(g) != 1:
            raise SyntaxError('addtitle error on line %d' % f.linenum)

          title = g[0]

        elif b.startswith('noeqs'):
          f.eqs = False

        elif b.startswith('noeqcache'):
          f.eqcache = False

        elif b.startswith('eqsize'):
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          g = re.findall(r, b)
          if len(g) != 1:
            raise SyntaxError('eqsize error on line %d' % f.linenum)

          f.eqdpi = int(g[0])

        elif b.startswith('eqdir'):
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          g = re.findall(r, b)
          if len(g) != 1:
            raise SyntaxError('eqdir error on line %d' % f.linenum)

          f.eqdir = g[0]

  f.simple_footer = showfooter
  f.simple_css = not nodefaultcss
  f.simple_source = showsourcelink

  # Get the file started with the firstbit.
  out(f.outf, f.conf['firstbit'])

  if not nodefaultcss:
    out(f.outf, f.conf['defaultcss'])

  # Add per-file css lines here.
  for i in range(len(css)):
    if '.css' not in css[i]:
      css[i] += '.css'

  for x in css:
    hb(f.outf, f.conf['specificcss'], x)

  for x in js:
    hb(f.outf, f.conf['specificjs'], x)

  # Look for a title.
  if pc(f) == '=': # don't check exact number f.outf '=' here jem.
    t = br(nl(f), f)[:-1]
    if title is None:
      title = re.sub(' *(<br />)|(&nbsp;) *', ' ', t)
  else:
    t = None

  #if title:
  hb(f.outf, f.conf['windowtitle'], title)

  out(f.outf, f.conf['bodystart'])


  if f.analytics:
    hb(f.outf, f.conf['analytics'], f.analytics)

  if fwtitle:
    out(f.outf, f.conf['fwtitlestart'])
    inserttitle(f, t)
    out(f.outf, f.conf['fwtitleend'])

  if menu:
    out(f.outf, f.conf['menustart'])
    insertmenuitems(*menu)
    
    # Inject the Top link directly into the bottom of the left column
    out(f.outf, '<div id="back-to-top"><a href="#">&uarr; Back to top</a></div>\n')
    
    out(f.outf, f.conf['menuend'])
  else:
    out(f.outf, f.conf['nomenu'])

  if not fwtitle:
    inserttitle(f, t)

  infoblock = False
  imgblock = False
  tableblock = False
  while 1: # wait for EOF.
    p = pc(f)

    if p == '':
      break

    elif p == '\\(':
      if not (f.eqs and f.eqsupport):
        break

      s = nl(f)
      # Quickly pull out the equation here:
      # Check we don't already have the terminating character in a whole-line
      # equation without linebreaks, eg \( Ax=b \):
      if not s.strip().endswith(r'\)'):
        while True:
          l = nl(f, codemode=True)
          if not l:
            break
          s += l
          if l.strip() == r'\)':
            break
      out(f.outf, br(s.strip(), f))

    # look for lists.
    elif p == '-':
      dashlist(f, False)

    elif p == '.':
      dashlist(f, True)

    elif p == ':':
      colonlist(f)

    # look for titles.
       # look for titles.
    elif p == '=':
      (s, c) = nl(f, True)
      # trim trailing \n.
      s = s[:-1]
      
      # Generate a clean anchor ID from heading text
      anchor_id = s.lower().strip()
      anchor_id = re.sub(r'[^a-z0-9\s-]', '', anchor_id)
      anchor_id = re.sub(r'\s+', '-', anchor_id)
      
      hb(f.outf, '<h%d id="%s">|</h%d>\n' % (c, anchor_id, c), br(s, f))
    # look for comments.
    elif p == '#':
      l = nl(f)

    elif p in ('\n', '\r'):
      nl(f)

    # look for blocks.
    elif p == '~':
      nl(f)
      if infoblock:
        out(f.outf, f.conf['infoblockend'])
        infoblock = False
        nl(f)
        continue
      elif imgblock:
        out(f.outf, '</td></tr></table>\n')
        imgblock = False
        nl(f)
        continue
      elif tableblock:
        out(f.outf, '</td></tr></table>\n')
        tableblock = False
        nl(f)
        continue
      else:
        if pc(f) == '{':
          l = allreplace(nl(f))
          r = re.compile(r'(?<!\\){(.*?)(?<!\\)}', re.M + re.S)
          g = re.findall(r, l)
        else:
          g = []

        # process jemdoc markup in titles.
        if len(g) >= 1:
          g[0] = br(g[0], f)

        if len(g) in (0, 1): # info block.
          out(f.outf, f.conf['infoblock'])
          infoblock = True
          
          if len(g) == 1: # info block.
            hb(f.outf, f.conf['blocktitle'], g[0])

          out(f.outf, f.conf['infoblockcontent'])

        elif len(g) >= 2 and g[1] == 'table':
          # handles
          # {title}{table}{name}
          # one | two ||
          # three | four ||
          name = ''
          if len(g) >= 3 and g[2]:
            name += ' id="%s"' % g[2]
          out(f.outf, '<table%s>\n<tr class="r1"><td class="c1">' % name)
          f.tablerow = 1
          f.tablecol = 1

          tableblock = True

        elif len(g) == 2:
          codeblock(f, g)

        elif len(g) >= 4 and g[1] == 'img_left':
          # handles
          # {}{img_left}{source}{alttext}{width}{height}{linktarget}.
          g += ['']*(7 - len(g))
          
          if g[4].isdigit():
            g[4] += 'px'

          if g[5].isdigit():
            g[5] += 'px'

          out(f.outf, '<table class="imgtable"><tr><td>\n')
          if g[6]:
            out(f.outf, '<a href="%s">' % g[6])
          out(f.outf, '<img src="%s"' % g[2])
          out(f.outf, ' alt="%s"' % g[3])
          if g[4]:
            out(f.outf, ' width="%s"' % g[4])
          if g[5]:
            out(f.outf, ' height="%s"' % g[5])
          out(f.outf, ' />')
          if g[6]:
            out(f.outf, '</a>')
          out(f.outf, '&nbsp;</td>\n<td align="left">')
          imgblock = True

        else:
          raise JandalError("couldn't handle block", f.linenum)

    else:
      s = br(np(f), f, tableblock)
      if s:
        if tableblock:
          hb(f.outf, '|\n', s)
        else:
          hb(f.outf, '<p>|</p>\n', s)

  if showfooter and (showlastupdated or showsourcelink):
    out(f.outf, f.conf['footerstart'])
    if showlastupdated:
      if showlastupdatedtime:
        ts = '%Y-%m-%d %H:%M:%S %Z'
      else:
        ts = '%Y-%m-%d'
      s = time.strftime(ts, time.localtime(time.time()))
      hb(f.outf, f.conf['lastupdated'], s)
    if showsourcelink:
      hb(f.outf, f.conf['sourcelink'], f.inname)
    out(f.outf, f.conf['footerend'])

  if menu:
    out(f.outf, f.conf['menulastbit'])
  else:
    out(f.outf, f.conf['nomenulastbit'])

  out(f.outf, f.conf['bodyend'])




SIMPLE_CSS = r"""
:root { color-scheme: light; --ink:#252b33; --muted:#606975; --blue:#314f78; --line:#e3e6ea; }
* { box-sizing:border-box; }
html { scroll-padding-top:calc(var(--header-height, 108px) + 20px); }
body { margin:0; background:#fff; color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; font-size:16px; line-height:1.65; -webkit-font-smoothing:antialiased; }
a { color:var(--blue); text-decoration:none; text-underline-offset:4px; }
a:hover { text-decoration:underline; }
.intro p a,.plain-list a { text-decoration:underline; text-decoration-color:#b2bfce; text-decoration-thickness:1px; }
a:focus-visible,summary:focus-visible,button:focus-visible { outline:2px solid var(--blue); outline-offset:5px; border-radius:2px; }
p { margin:0 0 14px; }
.page { max-width:880px; margin:auto; padding:0 38px; }
.site-header { position:sticky; top:0; z-index:5; background:rgba(255,255,255,.9); -webkit-backdrop-filter:blur(8px); backdrop-filter:blur(8px); padding:35px 0 24px; display:flex; align-items:baseline; justify-content:space-between; gap:20px; border-bottom:1px solid var(--line); }
.site-name { color:var(--ink); font-family:Georgia,"Times New Roman",serif; font-size:29px; font-weight:400; letter-spacing:-.8px; white-space:nowrap; }
.site-name a { color:inherit; }
nav { display:flex; gap:23px; font-size:14px; }
nav a { color:var(--muted); }
.intro { padding:36px 0 34px; }
h1 { font-size:24px; font-weight:500; line-height:1.3; letter-spacing:-.5px; margin:0 0 5px; }
.position { color:var(--muted); font-size:14px; margin-bottom:23px; }
.intro>p:not(.position) { max-width:72ch; }
.profile-links { display:flex; gap:8px 20px; flex-wrap:wrap; font-size:14px; margin-top:22px; }
.profile-links .email { margin-left:auto; }
main>section:not(.intro) { padding:30px 0 28px; border-top:1px solid var(--line); }
.section-heading { display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:8px 20px; margin-bottom:25px; }
h2 { margin:0; font-size:22px; font-family:Georgia,"Times New Roman",serif; font-weight:400; color:var(--ink); letter-spacing:-.25px; }
.small-link,.section-note { font-size:12px; }
.section-note { color:var(--muted); }
.publication-year { display:grid; grid-template-columns:54px minmax(0,1fr); column-gap:16px; }
.publication-year>.year { color:var(--muted); font-size:13px; font-weight:400; margin:2px 0 0; font-variant-numeric:tabular-nums; }
.publication-year>ul { list-style:none; padding:0; margin:0; }
.publication { margin:0 0 24px; }
.publication h4 { font-size:15.5px; font-weight:550; line-height:1.5; margin:0 0 5px; }
.publication p { font-size:13px; line-height:1.55; margin:0 0 3px; }
.authors { color:var(--muted); }
.venue { color:#4c5560; font-style:italic; }
.publication-links { display:flex; gap:0 15px; flex-wrap:wrap; align-items:baseline; font-size:12px; margin-top:7px; position:relative; }
.citation { display:contents; }
.citation>summary { display:inline; color:var(--blue); font-size:12px; list-style:none; cursor:pointer; }
.citation>summary::-webkit-details-marker { display:none; }
.citation>summary:after { content:' +'; }
.citation[open]>summary:after { content:' −'; }
.citation-content { width:100%; flex-basis:100%; margin-top:9px; padding:12px 14px; background:#f6f7f9; border:1px solid var(--line); border-radius:3px; }
.citation:not([open])>.citation-content { display:none; }
.citation pre { font-size:11.5px; margin:8px 0 0; white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.6; }
.copy-citation { font:inherit; border:1px solid #b7c3d1; border-radius:3px; background:white; color:var(--blue); padding:4px 8px; cursor:pointer; }
.copy-status { margin-left:10px; color:var(--muted); }
.section-intro { margin-top:-15px; margin-bottom:18px; font-size:13px; color:var(--muted); }
.courses { margin-bottom:19px; }
.course { border-bottom:1px solid var(--line); }
.course>summary,.course-plain { padding:11px 0; display:flex; align-items:baseline; gap:14px; font-size:14px; list-style:none; }
.course>summary { cursor:pointer; }
.course>summary::-webkit-details-marker { display:none; }
.course>summary:after { content:'+'; color:var(--muted); width:12px; flex-shrink:0; text-align:center; }
.course[open]>summary:after { content:'−'; }
.course-plain { padding-right:26px; }
.semester { margin-left:auto; color:var(--muted); font-size:12px; white-space:nowrap; }
.course-topics { padding:0 0 10px 18px; font-size:13px; color:var(--muted); }
.course-topics ul { padding-left:16px; margin:3px 0; }
.course-topics li p { margin:0 0 2px; }
.archive { margin-top:14px; }
.archive>summary { color:var(--blue); font-size:14px; cursor:pointer; width:fit-content; }
.archive>summary::marker { font-size:10px; }
.details-body { padding-top:13px; }
.plain-list { padding-left:18px; margin:0; font-size:14px; }
.plain-list li { padding-left:3px; margin-bottom:8px; }
.plain-list li::marker { color:#9099a5; font-size:10px; }
.plain-list p { margin:0; }
.plain-list b { font-weight:500; }
.subheading { font-size:15px; font-weight:550; margin:26px 0 14px; }
.section-heading+.subheading { margin-top:-2px; }
.contact-layout { display:grid; grid-template-columns:1fr 1fr; gap:25px; font-size:14px; }
.contact-layout p { margin-top:12px; }
.contact-email { font-size:15px; }
.contact-email-image { display:block; position:relative; height:1.65em; }
.contact-email-image img { display:block; position:absolute; top:3.5px; left:0; height:auto; max-width:100%; margin:0!important; user-select:none; -webkit-user-select:none; -webkit-user-drag:none; }
address { font-style:normal; color:var(--muted); }
address a { display:inline-block; margin-top:8px; }
footer { border-top:1px solid var(--line); display:flex; justify-content:space-between; gap:20px; padding:24px 0 32px; color:var(--muted); font-size:12px; }
.skip-link { position:absolute; left:16px; top:-80px; background:white; padding:8px 12px; border:1px solid var(--blue); z-index:10; }
.skip-link:focus { top:10px; }
@media (max-width:660px) {
 html { scroll-padding-top:calc(var(--header-height, 140px) + 16px); }
 .page { padding:0 23px; }
 .site-header { flex-direction:column; gap:11px; padding:24px 0 20px; }
 .site-name { font-size:28px; }
 nav { width:100%; justify-content:space-between; gap:12px; font-size:13px; }
 .intro { padding:27px 0; }
 body { font-size:15px; }
 h1 { font-size:22px; }
 .profile-links { gap:9px 19px; }
 .profile-links .email { margin-left:0; flex-basis:100%; }
 .section-heading { margin-bottom:21px; }
 .publication-year { grid-template-columns:1fr; }
 .publication-year>.year { margin:0 0 11px; font-weight:600; }
 .publication h4 { font-size:15px; }
 .course>summary,.course-plain { flex-wrap:wrap; gap:2px 8px; position:relative; padding-right:23px; }
 .course>summary>span:first-child,.course-plain>span:first-child { flex-basis:100%; }
 .course>summary:after { position:absolute; right:0; top:13px; }
 .semester { margin-left:0; }
 .contact-layout { grid-template-columns:1fr; gap:8px; }
 .section-note { font-size:12px; }
}
@media (prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } }
@media print {
 .site-header { position:static; background:#fff; -webkit-backdrop-filter:none; backdrop-filter:none; }
 .page { max-width:none; padding:0; }
 body { font-size:10pt; }
 nav,.skip-link,footer,.copy-citation { display:none!important; }
 .publication,.course,.plain-list li { break-inside:avoid; }
 a { color:inherit; }
 .citation { display:none; }
}

/* Layout for the existing source's content and generic jemdoc features. */
.intro-subheading { font:500 13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; color:var(--muted); margin:20px 0 7px; }
.intro ul { padding-left:19px; margin:0 0 18px; font-size:14px; }
.intro li p { margin:0; }
.intro .profile-links { margin-top:20px; }
.course-institution { color:var(--muted); font-weight:400; font-size:12px; }
section>ul,.details-body>ul { padding-left:18px; font-size:14px; }
section>ul p,.details-body>ul p { margin:0 0 8px; }
img { max-width:100%; height:auto; }
table { border-collapse:collapse; max-width:100%; margin:18px 0; }
td,th { border:1px solid var(--line); padding:6px 10px; text-align:left; }
.imgtable td { border:0; }
.codeblock,.infoblock { margin:18px 0; }
.blocktitle { font-weight:550; margin-bottom:8px; }
.blockcontent { padding:14px; background:#f6f7f9; border:1px solid var(--line); overflow-x:auto; }
pre { overflow-x:auto; font-size:12px; }
main h3 { scroll-margin-top:28px; }
@media print { details:not(.citation)>* { display:block!important; } }

"""

SIMPLE_JS = r"""
const siteHeader = document.querySelector('.site-header');
if (siteHeader) {
  const updateHeaderHeight = () => {
    document.documentElement.style.setProperty('--header-height', `${siteHeader.getBoundingClientRect().height}px`);
  };
  updateHeaderHeight();
  if (window.ResizeObserver) new ResizeObserver(updateHeaderHeight).observe(siteHeader);
  else window.addEventListener('resize', updateHeaderHeight);
}

document.querySelectorAll('.copy-citation').forEach((button) => {
  if (!navigator.clipboard?.writeText) return;
  button.hidden = false;
  button.addEventListener('click', async () => {
    const content = button.closest('.citation-content');
    const status = content.querySelector('.copy-status');
    try {
      await navigator.clipboard.writeText(content.querySelector('code').textContent);
      status.textContent = 'Copied';
    } catch {
      status.textContent = 'Select and copy the citation below.';
    }
  });
});

function revealHashTarget() {
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  const target = document.getElementById(id);
  if (!target) return;
  let node = target;
  while (node) { if (node.tagName === 'DETAILS') node.open = true; node = node.parentElement; }
}
window.addEventListener('hashchange', revealHashTarget);
revealHashTarget();

"""

# Presentation layer for the copied jemdoc parser. No site content lives here.
from html import escape as html_escape, unescape as html_unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import base64
import mimetypes


def resolve_source_path(f, filename):
    """Resolve an include relative to the file containing the directive."""
    path = Path(filename)
    if path.is_absolute():
        return str(path)
    current = getattr(f.inf, 'name', None)
    base = Path(current).resolve().parent if current else Path(f.inname).resolve().parent
    return str(base / path)


def render_publications(records, filename, content):
    """Render already-parsed BibTeX data, preserving the exact citation text."""
    esc = html_escape
    download = 'data:application/x-bibtex;base64,' + base64.b64encode(content.encode('utf-8')).decode('ascii')
    parts = ['<div class="bibliography">',
             '<a class="bibliography-download small-link" href="%s" download="%s">Download bibliography ↓</a>'
             % (download, esc(Path(filename).name, quote=True))]
    years = sorted({r['sort_year'] for r in records}, reverse=True)
    for year in years:
        parts.append('<div class="publication-year"><h3 class="year">%s</h3><ul>' % (year or 'Undated'))
        group = sorted((r for r in records if r['sort_year'] == year), key=lambda r: r['category'])
        for r in group:
            parts.append('<li class="publication"><h4>%s</h4>' % esc(r['title']))
            if r['author']:
                parts.append('<p class="authors">%s</p>' % esc(r['author']))
            if r['venue']:
                parts.append('<p class="venue">%s</p>' % esc(r['venue']))
            parts.append('<div class="publication-links">')
            for label, url in r['links']:
                parts.append('<a href="%s">%s</a>' % (esc(url, quote=True), esc(label)))
            if r['raw']:
                parts.append('<details class="citation"><summary>BibTeX</summary>'
                             '<div class="citation-content"><button class="copy-citation" type="button" hidden>Copy citation</button>'
                             '<span class="copy-status" role="status" aria-live="polite"></span>'
                             '<pre><code>%s</code></pre></div></details>' % esc(r['raw']))
            parts.append('</div></li>')
        parts.append('</ul></div>')
    parts.append('</div>')
    return '\n'.join(parts)


class HtmlNode:
    """Small HTML tree: keeps arbitrary jemdoc blocks rather than re-parsing prose."""
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, tag='', attrs=(), children=None):
        self.tag, self.attrs = tag, dict(attrs)
        self.children = list(children or [])

    def find(self, tag=None, **attrs):
        return next((n for n in self.walk() if (tag is None or n.tag == tag)
                     and all(n.attrs.get(k) == v for k, v in attrs.items())), None)

    def walk(self):
        for child in self.children:
            if isinstance(child, HtmlNode):
                yield child
                yield from child.walk()

    def text(self):
        return ''.join(c.text() if isinstance(c, HtmlNode) else c for c in self.children)

    def inner(self):
        return ''.join(c.render() if isinstance(c, HtmlNode) else
                       (c if self.tag in ('script', 'style', '!raw') else html_escape(c))
                       for c in self.children)

    def render(self):
        if self.tag in ('', '!raw'):
            return self.inner()
        attrs = ''.join(' ' + k + ('' if v is None else '="' + html_escape(v, quote=True) + '"')
                        for k, v in self.attrs.items())
        opening = '<' + self.tag + attrs + '>'
        return opening if self.tag in self.VOID else opening + self.inner() + '</' + self.tag + '>'


class HtmlTree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = HtmlNode(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in HtmlNode.VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)

    def handle_comment(self, data):
        self.stack[-1].children.append(HtmlNode('!raw', children=['<!--' + data + '-->']))


def render_nodes(nodes):
    return ''.join(n.render() if isinstance(n, HtmlNode) else html_escape(n) for n in nodes)


def meaningful(nodes):
    return [n for n in nodes if isinstance(n, HtmlNode) or n.strip()]


def split_headings(nodes, level):
    """Return (heading, nodes) groups, including any leading unheaded content."""
    groups = []
    heading, content = None, []
    for node in nodes:
        if isinstance(node, HtmlNode) and node.tag == 'h' + str(level):
            groups.append((heading, content))
            heading, content = node, []
        else:
            content.append(node)
    groups.append((heading, content))
    return groups


def local_asset_url(value, base):
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path or value.startswith('#'):
        return value
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        return value
    mime = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii') + (
        '#' + parsed.fragment if parsed.fragment else '')


def embed_fragment_assets(fragment, base):
    """Resolve media before the reader leaves an included source file."""
    def tag(match):
        return re.sub(r'\b(src|poster)=(\x22|\x27)(.*?)\2',
                      lambda attr: attr[1] + '=' + attr[2] + html_escape(
                          local_asset_url(html_unescape(attr[3]), Path(base)), quote=True) + attr[2],
                      match[0], flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r'<(?:img|source|audio|video)\b[^>]*>', tag, fragment, flags=re.IGNORECASE | re.DOTALL)


def embed_assets(tree, base):
    """Keep the generated file portable, including the source's e-mail image."""
    for node in tree.walk():
        if node.tag in ('img', 'source', 'audio', 'video') and node.attrs.get('src'):
            node.attrs['src'] = local_asset_url(node.attrs['src'], base)
        if node.attrs.get('poster'):
            node.attrs['poster'] = local_asset_url(node.attrs['poster'], base)
        if node.tag == 'link' and node.attrs.get('rel') == 'stylesheet':
            url = node.attrs.get('href', '')
            parsed = urlsplit(url)
            path = base / unquote(parsed.path)
            if not parsed.scheme and path.is_file():
                css = path.read_text(encoding='utf-8')
                css = re.sub(r'url\([\s\x22\x27]*([^\x22\x27)]+)[\x22\x27\s]*\)',
                             lambda m: 'url("' + local_asset_url(m.group(1).strip(), path.parent) + '")', css)
                node.tag, node.attrs, node.children = 'style', {}, [css]
        if node.tag == 'script' and node.attrs.get('src'):
            parsed = urlsplit(node.attrs['src'])
            path = base / unquote(parsed.path)
            if not parsed.scheme and path.is_file():
                node.attrs.pop('src')
                node.children = [path.read_text(encoding='utf-8').replace('</script', '<\\/script')]


def disclosure(title, content, css='archive', anchor=''):
    id_attr = ' id="%s"' % html_escape(anchor, quote=True) if anchor else ''
    return '<details class="%s"%s><summary>%s</summary><div class="details-body">%s</div></details>' % (
        css, id_attr, title, content)


def render_courses(node):
    parts = ['<div class="courses">']
    for item in node.children:
        if not isinstance(item, HtmlNode) or item.tag != 'li':
            continue
        children = meaningful(item.children)
        paragraph = next((c for c in children if isinstance(c, HtmlNode) and c.tag == 'p'), None)
        if paragraph is None:
            parts.append(item.render())
            continue
        label = paragraph.inner()
        # Pull a semester off the end without fixing course names or institutions.
        match = re.match(r'^(.*?),\s*([^<>]+?),\s*((?:Fall|Spring|Summer|Winter|Autumn)\s+\d{4})\.?\s*$', label)
        if match:
            label = '<span>%s<span class="course-institution"> · %s</span></span><span class="semester">%s</span>' % match.groups()
        else:
            label = '<span>' + label + '</span>'
        remainder = [c for c in children if c is not paragraph]
        if remainder:
            parts.append('<details class="course"><summary>%s</summary><div class="course-topics">%s</div></details>' % (
                label, render_nodes(remainder)))
        else:
            parts.append('<div class="course course-plain">%s</div>' % label)
    parts.append('</div>')
    return ''.join(parts)


def render_section_content(nodes, section_id):
    result = []
    for heading, group in split_headings(nodes, 2):
        if heading is None:
            for node in group:
                if section_id == 'teaching' and isinstance(node, HtmlNode) and node.tag == 'ul':
                    result.append(render_courses(node))
                else:
                    result.append(render_nodes([node]))
            continue
        key = heading.attrs.get('id', '')
        label = heading.inner().rstrip(':')
        if section_id == 'teaching' and key in ('past-lectures', 'previous-teaching', 'student-interns', 'student-supervision'):
            labels = {'past-lectures': 'Previous teaching', 'student-interns': 'Student supervision'}
            result.append(disclosure(labels.get(key, label), render_nodes(group), anchor=key))
            continue
        heading.tag = 'h3'
        heading.attrs['class'] = 'subheading'
        result.append(heading.render())
        if section_id == 'activities' and key in ('invited-talks', 'committees'):
            for node in group:
                if isinstance(node, HtmlNode) and node.tag == 'ul':
                    entries = [c for c in node.children if isinstance(c, HtmlNode) and c.tag == 'li']
                    if len(entries) > 3:
                        recent = HtmlNode('ul', {'class': 'plain-list'}, entries[:3])
                        previous = HtmlNode('ul', {'class': 'plain-list'}, entries[3:])
                        result.append(recent.render())
                        result.append(disclosure('Earlier talks' if key == 'invited-talks' else 'Previous committees &amp; service', previous.render()))
                        continue
                result.append(render_nodes([node]))
        else:
            result.append(render_nodes(group))
    return ''.join(result)


def render_intro(nodes):
    result = []
    for node in nodes:
        if isinstance(node, HtmlNode) and node.tag == 'p':
            anchors = [c for c in node.children if isinstance(c, HtmlNode) and c.tag == 'a']
            other = ''.join(c.text() if isinstance(c, HtmlNode) and c.tag not in ('a', 'img')
                            else c if isinstance(c, str) else '' for c in node.children)
            if len(anchors) > 1 and not other.strip(' \n\t,;·|'):
                node.tag = 'div'
                node.attrs = {'class': 'profile-links', 'aria-label': 'Academic profiles'}
                node.children = anchors
                labels = {'hal': 'HAL', 'orcid': 'ORCID', 'google scholar': 'Google Scholar', 'github': 'GitHub'}
                for link in anchors:
                    if link.text().lower() in labels:
                        link.children = [labels[link.text().lower()]]
        if isinstance(node, HtmlNode) and node.tag == 'h2':
            node.attrs['class'] = 'intro-subheading'
        result.append(render_nodes([node]))
    return ''.join(result)


def render_contact(nodes):
    # Arrange source content like the proposal: e-mail above lab/office on the
    # left; postal address and map together on the right. Keep mail as an image.
    groups = meaningful(nodes)
    if groups and isinstance(groups[0], HtmlNode) and groups[0].tag == 'p':
        first = groups[0]
        lines, current = [], []
        for child in first.children:
            if isinstance(child, HtmlNode) and child.tag == 'br':
                lines.append(current); current = []
            else:
                current.append(child)
        if meaningful(current):
            lines.append(current)
        if len(lines) > 2:
            email = []
            address_links = []
            for item in groups[1:]:
                image = item.find('img') if isinstance(item, HtmlNode) else None
                if image and image.attrs.get('alt', '').lower() in ('mail', 'email', 'e-mail'):
                    email.append('<span class="contact-email contact-email-image">' + item.inner() + '</span>')
                else:
                    address_links.append(item.inner() if isinstance(item, HtmlNode) and item.tag == 'p' else render_nodes([item]))
            left = ''.join(email) + '<p>' + '<br>'.join(render_nodes(line) for line in lines[:2]) + '</p>'
            right = '<address>' + '<br>'.join(render_nodes(line) for line in lines[2:])
            if address_links:
                right += '<br>' + '<br>'.join(address_links)
            right += '</address>'
            return '<div class="contact-layout"><div>' + left + '</div>' + right + '</div>'
    return '<div class="contact-content">' + render_nodes(nodes) + '</div>'


def render_simplified_page(document, f):
    parser = HtmlTree()
    parser.feed(document)
    tree = parser.root
    head = tree.find('head')
    container = tree.find(id='layout-content')
    if head is None or container is None:
        # A fully custom configuration may replace jemdoc's layout containers.
        return document
    base = Path(f.inname).resolve().parent
    # The source currently requests jemdoc.css twice. Replace just that legacy
    # theme; retain custom styles/scripts and all ordinary jemdoc content.
    head.children = [n for n in head.children if not (
        isinstance(n, HtmlNode) and n.tag == 'link' and
        Path(urlsplit(n.attrs.get('href', '')).path).name == 'jemdoc.css')]
    embed_assets(tree, base)
    titleblock = tree.find(id='toptitle')
    title = titleblock.find('h1') if titleblock else None
    title_html = title.inner() if title else 'Home'
    original_footer = container.find(id='footer')
    nodes = [n for n in container.children if n is not titleblock and not (
        isinstance(n, HtmlNode) and n.attrs.get('id') == 'footer')]
    if titleblock:
        subtitle = titleblock.find(id='subtitle')
        if subtitle:
            nodes.insert(0, subtitle)
    groups = split_headings(nodes, 1)
    intro, sections, contact = [], [], None
    for heading, content in groups:
        if heading is None:
            for subheading, subcontent in split_headings(content, 2):
                if subheading and subheading.attrs.get('id') == 'contact':
                    contact = (subheading, subcontent)
                    sections.append(contact)
                else:
                    if subheading: intro.append(subheading)
                    intro.extend(subcontent)
        elif heading.attrs.get('id') == 'contact':
            contact = (heading, content)
            sections.append(contact)
        else:
            sections.append((heading, content))
    section_ids = {h.attrs.get('id') for h, _ in sections}
    menu = tree.find(id='layout-menu')
    links = []
    if menu:
        for node in menu.walk():
            if node.tag != 'a' or node.attrs.get('href') == '#':
                if node.tag == 'a' and node.text().strip().lower() == 'contact' and contact:
                    links.append(('<a href="#contact">Contact</a>', 'contact'))
                continue
            href = node.attrs.get('href', '')
            if node.text().strip().lower().endswith('back to top'):
                continue
            if href.startswith('#') and href[1:] in section_ids:
                target = next(h for h, _ in sections if h.attrs.get('id') == href[1:])
                node.children = target.children[:]
            node.attrs.pop('class', None)
            links.append((node.render(), href.lstrip('#')))
    if not links:
        links = [('<a href="#%s">%s</a>' % (html_escape(h.attrs.get('id', ''), quote=True), h.inner()), '') for h, _ in sections]
    source_link = ''
    if getattr(f, 'simple_source', False):
        data = base64.b64encode(Path(f.inname).read_bytes()).decode('ascii')
        source_link = '<a href="data:text/plain;base64,%s" download="%s">Source</a>' % (data, html_escape(Path(f.inname).name, quote=True))
    # Preserve content/scripts supplied by custom bodystart/bodyend templates.
    before, after, seen_layout = [], [], False
    original_body = tree.find('body')
    for node in original_body.children if original_body else []:
        if isinstance(node, HtmlNode) and (node is container or container in list(node.walk())):
            seen_layout = True
        elif isinstance(node, HtmlNode) and node.attrs.get('id') == 'fwtitle':
            continue
        else:
            (after if seen_layout else before).append(node)
    body = [render_nodes(before), '<a class="skip-link" href="#main">Skip to content</a><div class="page">',
            '<header class="site-header"><h1 class="site-name"><a href="#top">%s</a></h1>' % title_html,
            '<nav aria-label="Main navigation">%s</nav></header><main id="main">' % ''.join(x[0] for x in links),
            '<section class="intro" aria-label="About">%s</section>' % render_intro(intro)]
    for heading, content in sections:
        key = heading.attrs.get('id', '')
        heading.tag = 'h2'
        heading.attrs['id'] = key + '-title'
        extra = ''
        for node in content:
            if isinstance(node, HtmlNode):
                bibliography = node if node.attrs.get('class') == 'bibliography' else None
                if bibliography:
                    download = bibliography.find('a', **{'class': 'bibliography-download small-link'})
                    if download:
                        bibliography.children.remove(download)
                        extra += download.render()
        body.append('<section id="%s" aria-labelledby="%s-title"><div class="section-heading">%s%s</div>' % (
            html_escape(key, quote=True), html_escape(key, quote=True), heading.render(), extra))
        body.append(render_contact(content) if key == 'contact' else render_section_content(content, key))
        body.append('</section>')
    body.append('</main>')
    if getattr(f, 'simple_footer', True):
        defaults = parseconf([])
        custom_footer = any(f.conf.get(key) != defaults.get(key)
                            for key in ('footerstart', 'footerend', 'lastupdated', 'sourcelink'))
        if original_footer and custom_footer:
            body.append(original_footer.render())
        body.append('<footer><span>%s</span>%s<a href="#top">Back to top ↑</a></footer>' % (title_html, source_link))
    body.append('</div>' + render_nodes(after))
    if getattr(f, 'simple_css', True):
        head.children.insert(0, HtmlNode('style', children=[SIMPLE_CSS]))
    # HTML5, responsive metadata and progressive enhancement, all self-contained.
    return '<!doctype html>\n<html lang="en"><head>\n<meta name="viewport" content="width=device-width,initial-scale=1">\n' + head.inner() + '\n</head><body id="top">\n' + ''.join(body) + '\n<script>' + SIMPLE_JS + '</script>\n</body></html>\n'


def procfile(f):
    """Run the copied jemdoc engine, then apply the new presentation."""
    destination = f.outf
    buffer = io.StringIO()
    f.outf = buffer
    try:
        _procfile_jemdoc(f)
        rendered = render_simplified_page(buffer.getvalue(), f)
    finally:
        f.outf = destination
        buffer.close()
    destination.write(rendered)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', metavar='OUTPUT', help='HTML file or output directory (end a new directory with /)')
    parser.add_argument('-c', metavar='CONFIG', action='append', default=[], help='jemdoc configuration file')
    parser.add_argument('--show-config', action='store_true', help='print the jemdoc configuration')
    parser.add_argument('--version', action='store_true', help='print version information')
    parser.add_argument('sources', nargs='*', help='.jemdoc source files (extension may be omitted)')
    args = parser.parse_args()
    if args.show_config:
        print(standardconf())
        return
    if args.version:
        print('new_jemdoc 1.0 — based on the project’s jemdoc 0.7.3')
        return
    if not args.sources:
        parser.print_help()
        return
    try:
        conf = parseconf(args.c)
        destination = Path(args.o).expanduser() if args.o else None
        is_directory = destination is not None and (destination.is_dir() or args.o.endswith(('/', os.sep)))
        if len(args.sources) > 1 and destination is not None and not is_directory:
            parser.error('multiple input files require an output directory')
        if is_directory:
            destination.mkdir(parents=True, exist_ok=True)
        for name in args.sources:
            source = Path(name).expanduser()
            if not source.is_file() and source.suffix != '.jemdoc':
                source = Path(str(source) + '.jemdoc')
            source = source.resolve()
            output = (destination / (source.stem + '.html') if is_directory else destination) if destination else source.with_suffix('.html')
            if output.resolve() == source:
                parser.error('the output must not overwrite the input source')
            # Render completely before opening the destination: a missing
            # include/bibliography must not truncate an existing good page.
            with source.open('rb') as infile, io.StringIO() as buffer:
                control = controlstruct(infile, buffer, conf, str(source))
                procfile(control)
                result = buffer.getvalue()
            with output.open('w', encoding='utf-8') as outfile:
                outfile.write(result)
    except (OSError, JandalError, SyntaxError) as exc:
        parser.exit(1, 'new_jemdoc: %s\n' % exc)


if __name__ == '__main__':
    main()
