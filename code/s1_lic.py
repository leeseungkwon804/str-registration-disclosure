#!/usr/bin/env python3
"""Classification of the public registration field into five categories (rules fixed before classification). Never returns a field value."""
import re, html

BR_RE   = re.compile(r'<\s*br\s*/?\s*>', re.I)
TAG_RE  = re.compile(r'<[^>]*>')
WS_RE   = re.compile(r'\s+')
PUNCT_ONLY = re.compile(r'^[\s;,.\-_/\\|()\[\]]*$')
DIGIT_RE = re.compile(r'\d')

BLANK_WORDS = {'n/a','na','none','nil','null','-','--','null null','no license',
               'no licence','not applicable','nicht vorhanden','sin licencia'}

EXEMPT_PAT = ['exempt','exemption','not required','no registration required',
    'not subject to','outside the scope','does not require',
    'exento','exenta','no sujeto','no requiere',
    'esente','esenzione','non soggetto',
    'exoner','exonér','non soumis','dispense','dispensé',
    'befreit','befreiung','ausgenommen','nicht erforderlich','keine genehmigung',
    'isento','isenta','dispensado',
    'vrijgesteld','vrijstelling','niet vereist',
    'εξαίρεση','εξαιρείται',
    'available with a mobility lease']

PENDING_PAT = ['pending','in progress','applied','application','awaiting','requested',
    'in process','being processed','submitted',
    'en trámite','en tramite','tramitación','solicitado','en curso',
    'in attesa','richiesto','in corso','in fase di',
    'en cours','demande en cours','demandé','en attente',
    'in bearbeitung','beantragt','antrag',
    'em curso','pedido',
    'in behandeling','aangevraagd',
    'σε εξέλιξη','σε αναμονή']

SPLIT_RE = re.compile(r'[;,/|]')

def _len_bucket(n):
    if n == 0: return '0'
    if n <= 10: return '1-10'
    if n <= 30: return '11-30'
    if n <= 80: return '31-80'
    return '81+'

def classify(raw):
    """returns dict(lic_class, has_number, n_tokens, has_br, len_bucket). 원문은 반환하지 않는다."""
    # 1단계 진입 전: 결측 (None / float nan / pandas.NA 모두)
    missing = raw is None
    if not missing:
        try:
            missing = bool(raw != raw)      # float('nan')
        except (TypeError, ValueError):
            missing = True                  # pd.NA 등 진리값이 모호한 결측
    if missing:
        return dict(lic_class='blank', has_number=0, n_tokens=0, has_br=0, len_bucket='0')
    s = str(raw)
    has_br = 1 if re.search(r'<\s*br', s, re.I) else 0
    # 0단계
    s = BR_RE.sub(';', s)
    s = TAG_RE.sub('', s)
    s = html.unescape(s)
    s = s.replace(' ',' ').replace(' ',' ').replace(' ',' ')
    s = WS_RE.sub(' ', s).strip()
    low = s.lower()
    lb = _len_bucket(len(s))
    toks = [t.strip() for t in SPLIT_RE.split(s)]
    toks = [t for t in toks if t]
    n_tokens = len(toks)

    def out(c):
        return dict(lic_class=c, has_number=1 if c=='number' else 0,
                    n_tokens=n_tokens, has_br=has_br, len_bucket=lb)
    # 1단계 blank
    if len(s) == 0 or PUNCT_ONLY.match(s) or low in BLANK_WORDS:
        return dict(lic_class='blank', has_number=0, n_tokens=0, has_br=has_br, len_bucket=lb)
    # 2단계 exempt (pending 보다 먼저)
    for p in EXEMPT_PAT:
        if p in low: return out('exempt')
    if low == 'hotel': return out('exempt')
    # 3단계 pending
    for p in PENDING_PAT:
        if p in low: return out('pending')
    # 4단계 number
    for t in re.split(r'[\s;,|]+', s):
        if len(t) >= 3 and len(DIGIT_RE.findall(t)) >= 2:
            return out('number')
    # 5단계
    return out('text_other')

CLASSES = ['blank','exempt','pending','number','text_other']
