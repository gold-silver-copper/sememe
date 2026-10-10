# Meaning guide (version 1)

This guide tells the teacher how to write down what a sentence means, so that
separate calls agree. Meanings that disagree give many lexical entries per
word, none of which pays for itself. The learner keeps only what lowers the
total cost, so a wrong annotation simply fails to pay; a consistent one
teaches.

## The shape of a meaning

A meaning is a tree of **nodes**. Each node has:

- a **concept**: a type from the ontology, named in `UpperCamelCase` after
  the idea, not after an English word when the two differ (`Possess`, not
  `Have`; `Cause` not `Make`);
- optional **features** (`time=past`, `def=yes`): small closed sets of
  values, listed below;
- **role** children (`agent: ...`), which are nodes themselves;
- an **entity ID** (`e1`, `e2`, ...) that is the same for every mention of
  the same thing in the passage ("Lily ... she ... the girl" are all `e1`);
- optionally a **name** for proper names ("Lily"), lowercased;
- the **words** that express it (word indices, written as anchors `^i`).

Each sentence unit has one root node. Events and states are the usual roots.

## Language-neutral

Write what is meant, not how English says it. English-only distinctions
(articles, the English tense system, word order, auxiliary "do", infinitive
"to", agreement endings) are not concepts. Represent instead:

| Feature | Values | On | Meaning |
|---|---|---|---|
| `def` | `yes`, `no` | things | the hearer can identify the referent (`yes`) or it is new (`no`) |
| `num` | `one`, `many` | things | how many |
| `time` | `past`, `now`, `future` | events, states | when, relative to the moment of speaking |
| `aspect` | `done`, `ongoing`, `habit` | events | finished, in progress, habitual |
| `polarity` | `neg` | events, states, properties | the negation of the node |
| `mood` | `ask`, `order`, `wish`, `can`, `must` | events, states | a question, a command, a wish, ability or possibility, obligation |

Only add a feature when the sentence actually expresses it. Leave out what is
not expressed (an English bare present "cats eat fish" needs no `time`).

## Roles

Use these roles. Propose a new role only when none fits, with a domain and a
range.

| Role | Domain → range | For |
|---|---|---|
| `agent` | Event → Thing | who does it on purpose |
| `patient` | Event → Thing | what is affected, changed, made, moved |
| `theme` | Thing → Thing | what a state or property is about; what is located, possessed, perceived as a whole |
| `experiencer` | Event → Thing | who feels, perceives, knows, wants |
| `stimulus` | Event → Thing | what is felt, perceived, liked |
| `recipient` | Event → Thing | who receives |
| `source` | Thing → Thing | where from |
| `goal` | Thing → Thing | where to |
| `location` | Thing → Thing | where |
| `time` | Thing → Thing | when (a time expression; `time=` is the feature) |
| `manner` | Thing → Thing | how |
| `instrument` | Event → Thing | with what |
| `attr` | Thing → Thing | a property of a thing (`Big`, `Red`) |
| `possessor` | Thing → Thing | whose |
| `quantity` | Thing → Thing | how many or how much (a number concept, `Some`, `All`) |
| `content` | Event → Thing | what is said, thought, known, wanted (often an event) |
| `topic` | Thing → Thing | what something is about |
| `cause` | Thing → Thing | why |
| `purpose` | Event → Thing | what for |
| `companion` | Event → Thing | together with whom |
| `degree` | Thing → Thing | how much of a property (`Very`) |
| `op` | Thing → Thing | the parts of a coordination (`And`, `Or`, `But`) |

## Recurring cases

- **Pronouns and other references.** "I", "me", "my" refer to the speaker,
  "you" to the addressee: write `@speaker` or `@addressee`. "she", "it",
  "they", "this" refer to an entity already in the passage: write its ID
  alone, `e3`. If the referent is not in the passage, write `@other`.
- **"X is ADJ", "X is a N".** The property or class is the root, with the
  thing as `theme`: "the cat is big" = `Big(theme: Cat[def=yes])`; "it is a
  dog" = `Dog(theme: ref)`. Add `time=` to the root if tense is expressed.
- **Existence.** "there is a cat" = `Exist(theme: Cat[def=no])`.
- **Possession.** "my ball" = `Ball(possessor: @speaker)`; "Tom has a ball" =
  `Possess(theme: Ball[def=no], possessor: Tom)`... use `Possess` with
  `possessor` and `theme`.
- **Questions.** Yes/no question: `mood=ask` on the root. Wh-question: the
  questioned slot is a `Query` node, and the root
  gets `mood=ask`: "what did you eat?" = `Eat(agent: @addressee, patient:
  Query, time=past, mood=ask)`.
- **Commands.** `mood=order` on the root; the doer is `@addressee` if not
  said.
- **Negation.** `polarity=neg` on the node negated.
- **Coordination.** `And(op: A, op: B)`, `Or(...)`, `But(...)`.
- **Function words with no meaning of their own** (infinitive "to", auxiliary
  "do", complementizer "that", expletive "it", "of" in "a cup of tea" when it
  only links) are anchored to nothing.
- **Prepositions** express a role: anchor them to the edge they mark ("in the
  park" anchors "in" to the `location` edge).
- **Determiners and auxiliaries** express features: anchor "the" to `def=yes`
  of its noun, "will" to `time=future`, "did" (in "did you go") to
  `time=past` and `mood=ask`, "not" to `polarity=neg`.
- **Inflections** stay with their word: "cats" anchors the `Cat` node, which
  has `num=many`; "walked" anchors `Walk` with `time=past`.
- **Interjections and discourse words** are concepts: "yes" `Yes`, "no" `No`,
  "oh" `Oh`, "hello" `Greet`, "thank you" `Thank`, "okay" `Agree`. A
  one-word unit is just that node.
- **Numbers** are concepts: `Two`, `Ten`; "two cats" = `Cat[num=many]
  (quantity: Two)`.
- **Names** are a node of the right concept with a name: "Lily" =
  `Girl^0#e1"lily"`, or `Person` when nothing more is known.
- **Shared arguments** ("I want to go": who goes is the speaker): write the
  inner event without repeating the argument: `Want(experiencer: @speaker,
  content: Go)`.
- **Idioms and fixed phrases** whose meaning is not built from their parts
  ("kick the bucket") anchor all their words to one node.

## The ontology

Prefer a concept that already exists. Before inventing one, check the listed
concepts (and their docs) for one that means the same. A new concept needs a
parent (its supertype) and a one-line doc. Build a hierarchy: `Girl <
Person < Animate < Thing`, `Run < Move < Event`, `Big < Size < Property`.
Useful top levels: `Thing` (the root), `Event` (things that happen),
`State` (things that hold), `Property` (qualities), `Animate`, `Person`,
`Animal`, `Object`, `Place`, `Substance`, `Time`, `Quantity`, `Discourse`
(interjections and speech acts).

## Notation

Write each unit's meaning in this compact form (it is the program's own term
syntax plus word anchors and entity IDs):

```text
node     := head anchor? entity? name? feats? roles?
head     := Concept | Query | @speaker | @addressee | @other | eN   (eN alone: a pronoun-like reference to entity eN)
anchor   := ^i or ^i,j,...        the word indices that express this node, feature or role
entity   := #eN                   the entity ID (same for every mention of the same thing)
name     := "lowercased name"
feats    := [feature=value anchor?, ...]
roles    := (role anchor?: node, ...)
```

- An anchor on a feature (`def=yes^3`) names the word that expresses it
  ("the"). A feature without an anchor is expressed by the node's own word
  (an inflection: "walked", "cats") or by nothing.
- An anchor on a role (`location^3: ...`) names the word that marks the role
  (a preposition).
- One word can anchor several features (`time=past^0, mood=ask^0`).
- Words that express nothing are simply not anchored anywhere.

## Example

"did you see the big dog?" (words: 0 did, 1 you, 2 see, 3 the, 4 big, 5 dog):

```text
See^2#e1[time=past^0, mood=ask^0](experiencer: @addressee^1, stimulus: Dog^5#e2[def=yes^3](attr: Big^4#e3))
```

"she put it in the box" after "Lily found a ball" (Lily is e1, the ball e2):

```text
Put^1#e4[time=past](agent: e1^0, patient: e2^2, goal^3: Box^5#e5[def=yes^4])
```
