"""
Detection prompts for propaganda span annotation.

The BASELINE prompt is designed to be the strongest possible single-pass
detector, incorporating:
- Official SemEval-2020 Task 11 technique definitions (Da San Martino et al., 2020)
- Disambiguation guidance for overlapping techniques
- Hierarchical decision process derived from the SemEval annotation flowchart (Figure 8)
- Correct span boundary rules from the SemEval annotation guidelines
- JSON span-text output format (Kasner & Dušek, 2024)

Research grounding:
- Kasner et al. (2025): detailed guidelines are the single most important
  factor for LLM span annotation quality. P_noguide hurt both models tested.
- Kasner et al. (2025): zero-shot with guidelines outperforms few-shot and CoT.
- Hasanain et al. (2024): GPT-4 achieves 81.82% precision with good prompts,
  but span index generation is unreliable — use text copying + string matching.
- Kasner et al. (2025): the biggest error source is technique misclassification
  (soft-hard F1 delta = 0.218), NOT span finding. Disambiguation guidance
  directly targets this failure mode.
"""

# ─── Official SemEval Technique Definitions with Disambiguation ───────────────

TECHNIQUE_GUIDELINES = """
## Propaganda Techniques

You must identify spans using EXACTLY these 14 technique labels. Each definition
below is from the official SemEval-2020 Task 11 annotation guidelines.

### Techniques targeting EMOTIONS:

1. **Loaded_Language**
   Using specific words and phrases with strong emotional implications (either
   positive or negative) to influence an audience.
   Examples: "a lone lawmaker's childish shouting", "how stupid and petty things
   have become in Washington"
   IMPORTANT: This is the catch-all for emotional language. Use it ONLY when the
   emotional words do not fit a more specific technique below. If the emotional
   language also serves to label/attack someone → use Name_Calling instead.
   If it appeals to group identity → use Flag-Waving instead.

2. **Name_Calling,Labeling**
   Labeling the object of the propaganda campaign as either something the target
   audience fears, hates, finds undesirable or loves, praises.
   Examples: "Republican congressweasels", "Bush the Lesser", "Public Enemy Number 1"
   NOTE: The label can be positive or negative. "Lesser" is pejorative even though
   it does not refer to "the second" — it is pejorative labeling.

3. **Appeal_to_Fear-Prejudice**
   Seeking to build support for an idea by instilling anxiety and/or panic in
   the population towards an alternative. In some cases the support is built
   based on preconceived judgements.
   Examples: "either we go to war or we will perish",
   "we must stop those refugees as they are terrorists"
   NOTE: This technique uses fear AS the argument. If the text merely describes
   a genuinely dangerous situation factually, that is NOT this technique.

4. **Flag-Waving**
   Playing on strong national feeling (or to any group; e.g., race, gender,
   political preference) to justify or promote an action or idea.
   Examples: "patriotism means no questions" (also a Slogan),
   "entering this war will make us have a better future in our country"
   NOTE: Must appeal to GROUP IDENTITY to justify a POSITION. A mere mention
   of a country or group is NOT flag-waving.

5. **Slogans**
   A brief and striking phrase that may include labeling and stereotyping.
   Slogans tend to act as emotional appeals.
   Examples: "Make America great again!", "BUILD THE WALL!",
   "The more women at war... the sooner we win."
   NOTE: Slogans are SHORT, MEMORABLE, ACTION-ORIENTED phrases. A long
   argumentative sentence is not a slogan even if it is emotionally charged.

6. **Exaggeration,Minimisation**
   Either representing something in an excessive manner: making things larger,
   better, worse (e.g., "the best of the best", "quality guaranteed") or making
   something seem less important or smaller than it really is (e.g., saying
   that an insult was just a joke).
   Examples: "Democrats bolted as soon as Trump's speech ended in an apparent
   effort to signal they can't even stomach being in the same room as the
   president", "We're going to have unbelievable intelligence"

### Techniques targeting LOGIC/REASONING:

7. **Causal_Oversimplification**
   Assuming a single cause or reason when there are actually multiple causes
   for an issue. Includes transferring blame to one person or group of people
   without investigating the complexities of the issue.
   Examples: "If France had not declared war on Germany then World War II
   would have never happened", "The reason New Orleans was hit so hard was
   because of all the immoral people who live there"

8. **Black-and-White_Fallacy**
   Presenting two alternative options as the only possibilities, when in fact
   more possibilities exist. As an extreme case, tell the audience exactly what
   actions to take, eliminating any other possible choices (Dictatorship).
   Examples: "You must be a Republican or Democrat",
   "There is no alternative to war"

9. **Thought-terminating_Cliches**
   Words or phrases that discourage critical thought and meaningful discussion
   about a given topic. They are typically short, generic sentences that offer
   seemingly simple answers to complex questions or that distract attention
   away from other lines of thought.
   Examples: "It is what it is", "It's just common sense", "You gotta do what
   you gotta do", "Nobody's perfect", "It doesn't matter", "Mind your own business"
   NOTE: The phrase must SHUT DOWN further analysis at a point where deeper
   examination is warranted. A genuine summary or conclusion is NOT this technique.

10. **Appeal_to_Authority**
    Stating that a claim is true simply because a valid authority or expert on
    the issue said it was true, without any other supporting evidence offered.
    Includes the special case where the reference is not an authority or expert
    (Testimonial).
    Examples: "Richard Dawkins says evolution is true. Therefore, it's true.",
    "According to Serena Williams, our foreign policy is the best on Earth."
    NOTE: If the authority IS a relevant expert AND evidence is also presented,
    this is NOT appeal to authority.

11. **Doubt**
    Questioning the credibility of someone or something.
    Examples: "A candidate talks about his opponent and says: Is he ready to
    be the Mayor?", "Can the same be said for the Obama Administration?"
    NOTE: This is about CASTING DOUBT through insinuation, not about
    presenting counter-evidence or factual criticism.

12. **Whataboutism,Straw_Men,Red_Herring**
    A technique that attempts to discredit an opponent's position by charging
    them with hypocrisy without directly disproving their argument.
    NOTE: SemEval merges this with Straw Man (misrepresenting someone's position
    then refuting the misrepresentation) and Red Herring (introducing irrelevant
    material to divert attention). All three divert from the actual argument.
    Examples: "President Trump —who himself avoided military service— keeps
    beating the war drums over North Korea"

13. **Bandwagon,Reductio_ad_hitlerum**
    Bandwagon: Attempting to persuade the audience to join in because "everyone
    else is taking the same action."
    Reductio ad hitlerum: Persuading an audience to disapprove an action by
    suggesting it is popular with groups hated by the target audience.
    Examples: "90% of citizens support our initiative. You should.",
    "Do you know who else was doing that? Hitler!"

14. **Repetition**
    Repeating the same message over and over again so that the audience will
    eventually accept it.
    Examples: "I still have a dream. It is a dream deeply rooted in the
    American dream. I have a dream that one day..."
    NOTE: Each repeated instance should be annotated as a SEPARATE span.

## Decision Process for Ambiguous Cases

When a span could match multiple techniques, use this priority:
1. If the text LABELS or ATTACKS a person/group with a pejorative term → Name_Calling
2. If the text is a SHORT, MEMORABLE, ACTION phrase → Slogans
3. If the text appeals to GROUP IDENTITY to justify a position → Flag-Waving
4. If the text instills FEAR as the primary argument → Appeal_to_Fear
5. If the text SIMPLIFIES causation to a single factor → Causal_Oversimplification
6. If the text presents ONLY TWO OPTIONS → Black-and-White_Fallacy
7. If the text DEFLECTS criticism by pointing to opponent's hypocrisy → Whataboutism,Straw_Men,Red_Herring
8. If the text SHUTS DOWN discussion with a stock phrase → Thought-terminating
9. If the text cites an AUTHORITY as sole evidence → Appeal_to_Authority
10. If the text CASTS DOUBT through insinuation → Doubt
11. If the text REPEATS a message → Repetition
12. If the text EXAGGERATES or MINIMISES → Exaggeration
13. If the text appeals to POPULARITY or compares to hated figures → Bandwagon
14. If it uses emotional language but none of the above apply → Loaded_Language

## Span Boundary Rules

- Annotate the MINIMAL text span where the propaganda technique appears.
  Include enough context to identify the technique, but do not include
  surrounding non-propagandistic text.
- One text span = one propaganda technique. If the same words exhibit
  multiple techniques, create SEPARATE annotations with the same span text
  but different technique labels.
- If a technique spans multiple sentences, include all relevant sentences.
"""

# ─── Output Format Instructions ───────────────────────────────────────────────

OUTPUT_FORMAT = """
## Output Format

Output a JSON object with a single key "annotations" containing a list.
Each annotation has three fields:
- "text": the EXACT text span copied verbatim from the input article.
  We will use string matching to locate it, so copy precisely.
- "type": the technique label, exactly as listed above (e.g., "Loaded_Language",
  "Name_Calling,Labeling", "Appeal_to_Fear-Prejudice", etc.)
- "reason": one sentence explaining why this span is this technique.

If no propaganda techniques are found, return: {"annotations": []}

Example output:
{"annotations": [
  {"text": "the corrupt elite", "type": "Name_Calling,Labeling", "reason": "Pejorative label applied to discredit without engaging with substance"},
  {"text": "destroying our way of life", "type": "Appeal_to_Fear-Prejudice", "reason": "Instills fear about consequences without presenting evidence"}
]}
"""

# ─── Baseline Zero-Shot Prompt ────────────────────────────────────────────────

ZERO_SHOT_SYSTEM = """You are an expert annotator trained in identifying propaganda techniques in news articles, following the SemEval-2020 Task 11 annotation guidelines.

Your task is to read the article carefully and identify ALL text spans that employ propaganda techniques. For each span, classify it using exactly one of the 14 technique labels defined below.

""" + TECHNIQUE_GUIDELINES + OUTPUT_FORMAT

ZERO_SHOT_USER = """Carefully read the following news article and identify all propaganda technique spans. Output your annotations as JSON.

ARTICLE:
\"\"\"
{text}
\"\"\"
"""

# ─── ASV Stage 1 Detection Prompt ─────────────────────────────────────────────
# Same quality as baseline but with explicit high-recall instruction.
# The consolidation stage will handle refinement.

STAGE1_SYSTEM = """You are an expert annotator trained in identifying propaganda techniques in news articles, following the SemEval-2020 Task 11 annotation guidelines.

Your task is to read the article carefully and identify ALL text spans that employ propaganda techniques. Be THOROUGH — it is better to flag a borderline case than to miss a genuine technique. A later consolidation stage will refine your detections.

For each span, classify it using exactly one of the 14 technique labels defined below.

""" + TECHNIQUE_GUIDELINES + OUTPUT_FORMAT

STAGE1_USER = ZERO_SHOT_USER  # Same user template

# ─── Few-Shot Prompt ──────────────────────────────────────────────────────────
# Examples drawn from SemEval-2020 Task 11 training data (Table 1 of the paper).
# Selected to cover a mix of common and rare techniques, plus a clean example.

FEW_SHOT_EXAMPLES = """
## Annotation Examples

Example 1 (multiple techniques in one passage):
ARTICLE: "When the left made Linda Sarsour into its role model, it climbed into bed with the worst of the worst. The father of a missing 4-year-old Georgia boy was training children at a filthy New Mexico compound to commit school shootings, prosecutors alleged in court documents Wednesday."
ANNOTATIONS:
{"annotations": [
  {"text": "its role model", "type": "Name_Calling,Labeling", "reason": "Labels Sarsour with a dismissive characterization"},
  {"text": "climbed into bed", "type": "Loaded_Language", "reason": "Emotionally charged metaphor implying complicity"},
  {"text": "the worst of the worst", "type": "Exaggeration,Minimisation", "reason": "Extreme exaggeration to characterise the subject"},
  {"text": "filthy", "type": "Name_Calling,Labeling", "reason": "Pejorative label applied to the compound"}
]}

Example 2 (appeal to authority and fear):
ARTICLE: "Monsignor Jean-François Lantheaume confirmed that 'Viganò said the truth. That's all.' A dark, impenetrable and 'irreversible' winter of persecution of the faithful by their own shepherds will fall."
ANNOTATIONS:
{"annotations": [
  {"text": "Monsignor Jean-François Lantheaume confirmed that 'Viganò said the truth. That's all.'", "type": "Appeal_to_Authority", "reason": "Uses a religious authority's statement as sole evidence for the claim"},
  {"text": "A dark, impenetrable and 'irreversible' winter of persecution of the faithful by their own shepherds will fall", "type": "Appeal_to_Fear-Prejudice", "reason": "Instills fear through apocalyptic imagery about religious persecution"}
]}

Example 3 (whataboutism):
ARTICLE: "President Trump —who himself avoided national military service in the 1960's— keeps beating the war drums over North Korea."
ANNOTATIONS:
{"annotations": [
  {"text": "President Trump —who himself avoided national military service in the 1960's— keeps beating the war drums over North Korea", "type": "Whataboutism,Straw_Men,Red_Herring", "reason": "Deflects from the current policy discussion by charging hypocrisy about military service"}
]}

Example 4 (no propaganda):
ARTICLE: "The committee released its quarterly findings on Tuesday, noting a 3% increase in operating costs compared to the previous quarter."
ANNOTATIONS:
{"annotations": []}
"""

FEW_SHOT_SYSTEM = ZERO_SHOT_SYSTEM + """

Here are examples of correct annotations from expert annotators to guide you:
""" + FEW_SHOT_EXAMPLES + """
Now annotate the following article using the same approach.
"""

FEW_SHOT_USER = ZERO_SHOT_USER