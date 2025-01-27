


# CLEVR Question Generation

CLEVR questions are generated using the script `generate_questions.py`, which is expected to be run from 
the `question_generation` directory.

This script reads a JSON file containing information about scenes (as produced by `render_images.py`) and outputs
a JSON file containing questions, functional programs, and answers for those images. In most cases the script will be invoked
like this:

```bash
python generate_questions.py --input_scene_file $INPUT_FILE --output_questions_file $OUTPUT_FILE
```

Question generation has no dependencies other than Python itself. The code was developed on Python 3.5, but should also
work on Python 2.7.

Questions are generated by instantiating **question templates**; the question templates used for our CVPR paper can be
found in the directory `CLEVR_1.0_templates`. Each file in this directory contains several related templates.

## Selecting input scenes
By default `generate_questions.py` will generate questions for all images in the input file. However you can generate questions 
for only a subset of images using the `--scene_start_idx` and `--num_scenes` flags: the former gives the index at which to
start generating questions, and the latter gives the number of images for which questions should be generated.
These flags can be useful for distributing question generation among many workers.

## Controlling questions per image
The flag `--templates_per_image` (default 10) is the number of templates that we will aim to instantiate for every image, and
the flag `--instances_per_template` gives the number of instantiations we will try to find per template. In total the number
of questions per image will be the product of `--templates_per_image` and `--instances_per_template`; however some images may
have slightly fewer questions if no valid template instantiations can be found.

## Question Templates
Each question template consists of four components:

1. One or more **parameters**, each with a type and a name. Instantiating the template amounts to choosing a value for
   each of these parameters; parameters may be given a `NULL` value
2. One or more **text templates** that give a natural-language representation of the question
3. A **program template** consisting of a sequence of **nodes**; each node in the program template
   may expand to multiple functions in the final program instantiated from the template
4. Zero or more **constraints** restricting the allowed values that the parameters are allowed to take.

Here is an example template:

```javascript
{
  "params": [
    {"type": "Size", "name": "<Z>"},
    {"type": "Color", "name": "<C>"},
    {"type": "Material", "name": "<M>"},
    {"type": "Shape", "name": "<S>"},
    {"type": "Relation", "name": "<R>"},
    {"type": "Size", "name": "<Z2>"},
    {"type": "Color", "name": "<C2>"},
    {"type": "Material", "name": "<M2>"},
    {"type": "Shape", "name": "<S2>"}
  ],
  "text": [
    "What size is the <Z2> <C2> <M2> <S2> [that is] <R> the <Z> <C> <M> <S>?",
    "What is the size of the <Z2> <C2> <M2> <S2> [that is] <R> the <Z> <C> <M> <S>?",
    "How big is the <Z2> <C2> <M2> <S2> [that is] <R> the <Z> <C> <M> <S>?",
    "There is a <Z2> <C2> <M2> <S2> [that is] <R> the <Z> <C> <M> <S>; what size is it?",
    "There is a <Z2> <C2> <M2> <S2> [that is] <R> the <Z> <C> <M> <S>; how big is it?",
    "There is a <Z2> <C2> <M2> <S2> [that is] <R> the <Z> <C> <M> <S>; what is its size?"
  ],
  "nodes": [
    {"type": "scene", "inputs": []},
    {"type": "filter_unique", "inputs": [0], "side_inputs": ["<Z>", "<C>", "<M>", "<S>"]},
    {"type": "relate_filter_unique", "inputs": [1], "side_inputs": ["<R>", "<Z2>", "<C2>", "<M2>", "<S2>"]},
    {"type": "query_size", "inputs": [2]}
  ],
  "constraints": [
    {"type": "NULL", "params": ["<Z2>"]}
  ]
}
```

### metadata.json
The special file `metadata.json` defines the simple functional programming language used to construct programs and
program templates.

### Template Parameters
Each template parameter has a type and a name; the allowed types are `Size`, `Color`, `Material`, `Shape`, and `Relation`.
The allowed values for each of these types is stored in `metadata.json`; in addition to the values defined here, each
non-`Relation` template parameter may also be assigned the value `NULL`.

By convention, `Size` parameters are called `<Z>`, `<Z2>`, `<Z2>`, etc; similarly `Color` parameters are called `<C>`, 
`Material` parameters are called `<M>`, `Shape` parameters are called `<S>`, and `Relation` parameters are called `<R>`.

### Text Templates
Each question template defines one or more **text templates** which give different ways of expressing the question in
natural language. Text templates must use all of the template parameters. After values have been chosen for all template
parameters, a natural language version of the question is generated by randomly choosing one of the text templates and
replacing the parameter names with their values. Parameters whose value is `NULL` are replaced with the empty string, unless
the parameter has type `Shape` in which case its textual value is `"thing"`.

To increase linguistic diversity, the file `synonyms.json` defines a set of synonyms for template parameter values,
e.g. `"ball"` is a synonym for `"sphere"`. When instantiating templates, values are randomly replaced by synonyms.

Text templates can also have optional segments; any text surrounded by brackets will be removed with probability 0.5 during
template instantiation. In the example above, the substring `"that is"` is optional in all text templates.

Finally, there are some special-case heuristics to replace the word `"other"` with `"another"`, `"a"`, or the empty string
in some circumstances to try and minimize ambiguity.

### Program Templates
A program template is defined as a sequence of nodes; each node receives input from zero or more other nodes, and produces
an output; this sequence is expected to be sorted topologically in the template. The inputs to each node are identified by
`nodes` field of a node, which is a list of integers indexing into the node sequence. A node in a program template may expand
to more than one node in the program instantiated from the template.

Each node has a **type**, such as `scene`
or `filter_color`; the `metadata.json` defines the full list of available nodes types, as well as input and output types for 
each node type.

In addition to receiving inputs from earlier nodes, some nodes also receive **side inputs** (also called **value inputs**
in some places); these are literal values of some type. The number and types of expected side inputs for all node types are
also listed in the `metadata.json` file.

As a concrete example, in the template above the first node has type `scene`; the `metadata.json` file gives us the following
information about this node type:

```javascript
// From metadata.json
{
  "name": "scene",
  "inputs": [],
  "output": "ObjectSet",
  "terminal": false
}
```

This indicates that `scene` nodes receive no inputs, and output an `ObjectSet`; `scene` nodes receive no side inputs, and
cannot be the final node in a fully instantiated program since they are not `terminal`.

The next node in the sequence above has type `filter_unique`; since its `input` is `[0]` it receives as input the output from 
the previous `scene` node. the `metadata.json` file gives us the following information about this node type:

```javascript
// From metadata.json
 {
   "name": "filter_unique",
   "inputs": ["ObjectSet"],
   "side_inputs": ["Size", "Color", "Material", "Shape"],
   "output": "Object",
   "terminal": false,
   "template_only": true
 }
 ```
Thus nodes of type `filter_unique` receive one input of type `ObjectSet` and four side inputs of type `Size`, `Color`, 
`Material`, and `Shape` (corresponding to parameters `<Z>`, `<C>`, `<M>`, `<S>` in the `side_inputs` field of the template 
node), and produce an output of type `Object`. Again, this node is not `terminal` so it cannot be the final node of a
fully instantiated program. This node type is marked as `template_only`, indicating this node type is only valid as part of
a program template and cannot be used in a fully instantiated program; during instantiation template nodes of type 
`filter_unique` will be replaced by a subsequence of `filter_size`, `filter_color`, `filter_material`, `filter_shape`, 
followed by a `unique` node. The use of special template-only nodes like this lead to more expressive templates, and also
allow us to more easily prune the search space during template instantiation.

Continuing with the example template above, the output from the `filter_unique` node is passed to another node of type
`relate_filter_unique`, which takes an input of type `Object` and five side inputs, and produces an output of type `Object`. 
This is another special template-only node type which will expand into a `relate` node followed by some subsequence of 
`filter_size`, `filter_color`, `filter_material`, `filter_shape`, followed by a `unique` node. The output
of the `relate_filter_unique` node is then passed to a node of type `query_size`, which takes an `Object` as input and
produces an output of type `Size`. This node type is terminal and is not template-only, so it will be the final node of both
the program template as well as all programs instantiated from that template.

### Constraints
Templates can define **constraints** on the values that template parameters are allowed to take; constraints can be necessary
to ensure that the question does not give away its answer. The example template above includes a constraint that the
parameter `<Z2>` must be `NULL`; without this constraint the template could produce questions such as *"What size is the big 
thing left of the sphere?"* which can be trivially answered from the text of the question.

The following two constraint types are supported:
- `NULL`: The parameter must take the value `NULL`, as in the example above.
- `OUT_NEQ`: The outputs of the two specified nodes must have different values when the instantiated program is run. This is used for templates like *"Are there an equal number of \<Z\> \<C\> \<M\> \<S\>s and \<Z2\> \<C2\> \<M2\> \<S2\>s?"* to ensure that the two question subparts refer to different sets of objects, which avoids trivial questions like *"Are there an equal number of spheres and balls?"*.
