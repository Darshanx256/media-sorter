import ast
import os
import gettext

localedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'locales')

def load_po(filepath):
    messages = {}
    if not os.path.exists(filepath):
        return messages
    msgid = None
    msgstr = None
    state = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('msgid '):
                if state == 2 and msgid is not None:
                    messages[msgid] = msgstr
                try:
                    msgid = ast.literal_eval(line[6:])
                    state = 1
                except (ValueError, SyntaxError):
                    pass
            elif line.startswith('msgstr '):
                try:
                    msgstr = ast.literal_eval(line[7:])
                    state = 2
                except (ValueError, SyntaxError):
                    pass
            elif line.startswith('"'):
                try:
                    s = ast.literal_eval(line)
                    if state == 1:
                        msgid += s
                    elif state == 2:
                        msgstr += s
                except (ValueError, SyntaxError):
                    pass
        if state == 2 and msgid is not None:
            messages[msgid] = msgstr
    return messages

try:
    translator = gettext.translation('messages', localedir=localedir, fallback=True)
    _default_gettext = translator.gettext
except Exception:
    def _default_gettext(s):
        return s

lang = os.environ.get('LANGUAGE') or os.environ.get('LC_ALL') or os.environ.get('LC_MESSAGES') or os.environ.get('LANG')
if lang:
    lang = lang.split('.')[0]

po_dict = {}
if lang:
    langs_to_try = [lang]
    if '_' in lang:
        langs_to_try.append(lang.split('_')[0])
        
    for l in langs_to_try:
        po_path = os.path.join(localedir, l, 'LC_MESSAGES', 'messages.po')
        if os.path.exists(po_path):
            po_dict = load_po(po_path)
            break

def _(s):
    if s in po_dict and po_dict[s]:
        return po_dict[s]
    return _default_gettext(s)
