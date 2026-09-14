"""Self-growing glossary of names and jargon learned across meetings.

Stores `{term: count}` in a JSON file (default `workspace/glossary_learned.json`,
gitignored like the speaker DB). Confirmed speaker names are pinned active
immediately; LLM-extracted entities only go active once they recur (count >=
min_count), so a one-off mistranscription never sticks.

The review `commit` calls `harvest()`, and `active_terms()` feeds the speaker
name typeahead — so naming a speaker or discussing a term today is remembered
for the next meeting with no manual editing.

These terms are deliberately NOT fed to Whisper as a vocabulary prompt. A
comma-separated list of proper nouns puts the model in list-continuation mode:
on low-information audio it continues the list instead of transcribing, which
destroyed real speech rather than merely decorating silence. Spelling is fixed
after the fact instead — see `correct_terms`.
"""

import json
import os
import re
from difflib import SequenceMatcher

from core import atomic_write

MIN_COUNT = 2

# Speaker-label artifacts the pipeline produces ("Unknown (SPEAKER_01)", the
# user's "Unknown Group 03" placeholders, "Me (Local)") are useless as
# vocabulary — nobody ever says those literal words.
_PARENTHETICAL_TAIL = re.compile(r"\s*\([^)]*\)\s*$")

# "Speaker 3" is a placeholder identity for someone not yet named, not a word
# anyone says.
_PLACEHOLDER_NAME = re.compile(r"^Speaker \d+$")


def clean_term(term):
    """Normalize a candidate glossary term, or return None to reject it.

    Strips a trailing parenthetical decoration ("HSF (Hippocampal Subfields)"
    -> "HSF") so the hint carries spoken words, and rejects speaker-label
    artifacts and too-short fragments.
    """
    t = _PARENTHETICAL_TAIL.sub("", (term or "").strip()).strip()
    if len(t) < 3:
        return None
    low = t.lower()
    if low.startswith("unknown") or low in ("me (local)", "me/local"):
        return None
    if _PLACEHOLDER_NAME.match(t):
        return None
    return t


# Longest glossary entry worth matching as a phrase ("Siemens Sensation 64").
# Beyond three words a near-miss is more likely to be a coincidence than a term.
_MAX_TERM_WORDS = 3

# Below this many characters a fuzzy match is meaningless: "pad"/"pet" and
# "tau"/"top" are one edit apart, so only exact-key matches may rewrite them.
_MIN_FUZZY_KEY = 6

# Similarity above which a single-word or phrase near-miss is corrected.
_FUZZY_CUTOFF = 0.85

# Ordinary English words are never rewritten by a fuzzy match, however close a
# glossary term looks. Whisper produces these constantly and a glossary of
# surnames and acronyms sits one or two edits from many of them ("Barr"/"bar",
# "Amelia"/"a media", "MAPE"/"map"). Restricted to the words that actually
# collide with this domain's vocabulary rather than a full dictionary.
_COMMON_WORDS = frozenset("""
a an as at be by do go he i if in is it me my no of on or so to up us we oh ok
yeah okay ant arm art age air aim add ago ah um uh
about above after again against all also and another any are around away back
because been before being below best better between both but came can come
could did does doing done down during each even every few first for from get
give goes going gone good got great had has have here how into its just keep
kind know large last later least less let like little long look made make many
maybe mean might more most much must near need never next not now off often
once only other our out over own part past per place point put rather really
right run said same say see seem set should show side since some soon still
such sure take than that the their them then there these they thing think this
those though three through time together too took top toward two under until
use used using very want was way well went were what when where which while
who why will with within without word work would year yet you your
bar bare bat bay bed bee bell belt bend best bet big bill bit bite black block
blue board boat body book born both box boy brain break bring broad brother
call car card care case cash cat catch cause cell chair chance change charge
check chest child choice church city claim class clean clear close cold color
cost count couple course court cover cream cross crowd cut dark data date day
dead deal death deep desk detail die dinner direct doctor dog door doubt draw
dream dress drink drive drop dry duty ear early earth east easy eat edge eight
else end enough enter equal error event exact eye face fact fail fair fall
family far fast fat father fear feel field fight figure file fill film final
find fine finger fire firm fish fit five fix flat floor flow fly focus follow
food foot force form four free fresh friend front full fun game gas gate girl
glass goal gold grade grand grass green ground group grow guard guess gun hair
half hall hand hang happy hard hat head health hear heart heat heavy help hide
high hill hit hold hole home hope horse hospital hot hotel hour house huge
human hundred hurt idea image inch index inside issue item join joy jump key
kid kill king kitchen knee knife lack lady land language late laugh law lay
lead leaf learn leave left leg lesson letter level lie life lift light limit
line lip list listen live load local lock lose loss lot loud love low luck
lunch machine main major man map march mark market marry mass master match
mate matter may meal measure meat media medical meet member memory mention
message metal method middle mile milk mind mine minute miss mix model modern
moment money month moon morning mother mount mouth move movie music name
narrow nation nature neck news night nine noise none normal north nose note
nothing notice number object ocean odd offer office oil old open opposite
order organ origin outside page pain paint pair paper parent park party pass
past path patient pattern pay peace pen people pepper perfect period person
pet phone photo phrase pick picture piece pink pipe pity plan plane plant
plastic plate play please plenty pocket police policy pool poor pop port
position possible post pound pour power practice press pretty price pride
prime print prize problem produce program project proof proper protect proud
prove public pull pure purple purpose push quality quarter queen question
quick quiet quite race radio rail rain raise range rank rapid rare rate reach
read ready real reason record red reduce refer reflect region regular relate
remain remember remove repeat reply report rest result return rich ride ring
rise risk river road rock role roll roof room root rope rose rough round row
rule safe sail salt sand save scale scene school science score sea search
season seat second secret section seed sell send sense sentence separate
serve service seven sex shall shape share sharp she sheet shelf shell shine
ship shirt shock shoe shoot shop short shot shoulder shout shut sick sight
sign silver simple simply sing single sir sister sit six size skill skin sky
sleep slide slip slow small smell smile smoke smooth snow social soft soil
sold soldier solid solve son song sorry sort sound source south space speak
special speed spell spend spirit spoke spot spread spring square stage stand
star start state station stay steal steam steel step stick stiff stock stone
stop store storm story straight strange street stress stretch strike string
strong study stuff style subject substance succeed such sudden suffer sugar
suggest suit summer sun supply support suppose surface surprise sweet swim
system table tail talk tall task taste tax teach team tear tell temperature
ten tend tent term test text thank thick thin third thirty thought thousand
thread threw throw thus tie tight till tiny tip tire title today toe told
tomorrow tone tongue tonight tool tooth total touch tour town track trade
train travel treat tree trip trouble truck true trust truth try tube turn
twelve twenty type ugly uncle understand union unit unless upon upper upset
value van various vast verb view village visit voice vote wage wait wake walk
wall war warm warn wash waste watch water wave weak wear weather week weight
welcome west wet wheel whether whole whom whose wide wife wild win wind window
wine wing winter wire wise wish woman wonder wood wool world worry worse worth
wrap write wrong yard yellow yes yesterday young
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


def _is_common(word):
    """Is this an ordinary English word? Checks the plural too, so the stoplist
    can stay a list of singulars ("ants" -> "ant")."""
    w = word.lower().strip("'’-.,")
    if w in _COMMON_WORDS:
        return True
    return w.endswith("s") and w[:-1] in _COMMON_WORDS


def _key(text):
    """Collapse to a comparison key: casefolded, punctuation and spacing gone.

    Lets "3D Slicer", "3DSlicer" and "3-D slicer" share one key, which is
    exactly the family of differences Whisper varies freely.
    """
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


class TermCorrector:
    """Restore canonical spellings of glossary terms after recognition.

    This replaces the old approach of priming Whisper with the term list, which
    made the model recite the glossary instead of transcribing. Correcting the
    output afterwards cannot invent text: the worst case is a word left alone.

    Two tiers, both matched on `_key`:

    1. Exact key -> canonical casing/punctuation ("freesurfer" -> "FreeSurfer",
       "3d slicer" -> "3D Slicer"). No judgement involved.
    2. Near miss above `cutoff` ("FeeSurfer" -> "FreeSurfer"), fenced in by a
       minimum key length, a common-word stoplist, and a rule against touching
       any span that is already some glossary term.

    Phonetically distant manglings stay put by design -- "Noah" scores poorly
    against "Avery", well under the cutoff. Rewriting those would need a
    phonetic model, and a wrong rewrite is worse than a wrong spelling.
    """

    def __init__(self, terms, cutoff=_FUZZY_CUTOFF):
        self.cutoff = cutoff
        self._canonical = {}          # key -> canonical spelling
        self._by_words = {}           # word count -> [(key, canonical)]
        for term in terms or ():
            c = clean_term(term)
            if not c:
                continue
            k = _key(c)
            if not k or k in self._canonical:
                continue
            n = len(c.split())
            if n > _MAX_TERM_WORDS:
                continue
            self._canonical[k] = c
            self._by_words.setdefault(n, []).append((k, c))
        self._max_words = max(self._by_words, default=0)

    def __bool__(self):
        return bool(self._canonical)

    def _match(self, span, n):
        """Canonical spelling for an n-word span, or None to leave it alone."""
        # A possessive is not part of the name. Without this "Amelia's paper"
        # keys as "amelias", fuzzy-matches "Amelia" at 0.92, and the correction
        # silently deletes the "'s".
        suffix = ""
        for poss in ("'s", "’s"):
            if span.endswith(poss) and len(span) > len(poss):
                span, suffix = span[: -len(poss)], poss
                break

        k = _key(span)
        if not k:
            return None
        words = span.split()
        first = words[0] if words else ""

        exact = self._canonical.get(k)
        if exact is not None:
            # An ordinary word that happens to spell an acronym ("ants" -> ANTs,
            # "pet" -> PET) is far more often the word. Losing that capitalisation
            # is cheap; capitalising the wrong "ants" reads as a mistake.
            if n == 1 and _is_common(first):
                return None
            return exact + suffix if exact + suffix != span + suffix else None

        if len(k) < _MIN_FUZZY_KEY:
            return None
        # A fuzzy match must not begin on an ordinary word. Without this, "in
        # 3dslicer" scores 0.89 against "3D Slicer" and swallows the "in", and
        # "a siemens sensation" eats the "a" then leaves "64" stranded. No
        # glossary phrase starts with an article or preposition.
        if _is_common(first):
            return None
        # Nor may it hinge on a single letter: "Virtual, P.D.:" scores 0.89
        # against "Virtual PBV" and becomes "Virtual PBV.D.:".
        if n > 1 and any(len(_key(w)) < 2 for w in words):
            return None
        best, best_ratio = None, self.cutoff
        for cand_key, canonical in self._by_words.get(n, ()):
            # Both ends have to survive. Whisper mangles the middle of a term,
            # rarely its first or last sound, so a match that rewrites an
            # endpoint is one that swallowed a neighbouring word: "V2
            # CT-emposema" -> "CT emphysema" loses the V2, and "CT-emposema
            # subtypes in" -> "CT emphysema subtypes" eats the "in". This costs
            # a handful of real fixes ("Wirtual" -> "Virtual", "Alexx" ->
            # "Alex") and prevents every corruption observed over three
            # meetings, which is the trade worth making.
            if k[0] != cand_key[0] or k[-1] != cand_key[-1]:
                continue
            r = SequenceMatcher(None, k, cand_key).ratio()
            if r > best_ratio:
                best, best_ratio = canonical, r
        return best + suffix if best else None

    def __call__(self, text):
        if not text or not self._canonical:
            return text
        tokens = list(_TOKEN_RE.finditer(text))
        out, i, cursor = [], 0, 0
        while i < len(tokens):
            hit = None
            # Longest phrase first, so "Virtual PET project" wins over "Virtual PET".
            for n in range(min(self._max_words, len(tokens) - i), 0, -1):
                span = text[tokens[i].start():tokens[i + n - 1].end()]
                replacement = self._match(span, n)
                if replacement is not None:
                    hit = (n, tokens[i].start(), tokens[i + n - 1].end(), replacement)
                    break
            if hit is None:
                i += 1
                continue
            n, start, end, replacement = hit
            out.append(text[cursor:start])
            out.append(replacement)
            cursor = end
            i += n
        if not out:
            return text
        out.append(text[cursor:])
        return "".join(out)


def _load(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except (OSError, ValueError):
            # Corrupt/unreadable store: preserve it aside for inspection rather
            # than silently clobbering it, then start fresh.
            try:
                os.replace(path, path + ".bad")
            except OSError:
                pass
    return {}


def _save(path, store):
    """Write the store atomically (temp + rename) so a crash mid-write can't
    truncate and erase every learned term."""
    atomic_write.write_text(path, json.dumps(store, indent=2))


def active_terms(path, min_count=MIN_COUNT):
    """Terms seen often enough to trust (confirmed names are pinned >= min_count),
    filtered through clean_term so pre-existing pollution in the store never
    reaches the ASR prompt even though the file is left untouched."""
    out = []
    for t, c in _load(path).items():
        if c >= min_count and clean_term(t) is not None:
            out.append(t)
    return out


def harvest(path, names=(), entities=(), min_count=MIN_COUNT):
    """Fold a finished meeting's terms into the store and return the active list.

    - names: confirmed speaker names -> pinned active immediately.
    - entities: LLM-extracted terms -> +1 each (active only after recurring).

    Both are sanitized via clean_term, so speaker-label junk never enters.
    """
    store = _load(path)
    for n in names:
        c = clean_term(n)
        if c:
            store[c] = max(store.get(c, 0), min_count)
    for e in entities:
        c = clean_term(e)
        if c:
            store[c] = store.get(c, 0) + 1
    _save(path, store)
    return [t for t, c in store.items() if c >= min_count and clean_term(t) is not None]
