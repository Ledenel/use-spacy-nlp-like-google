# python-zh-nlp
a docker image for python chinese nlp, packaged with common models from spacy

## Usage

Since load model can be time-consuming, it's better to use via a simple web service.

when you send raw text to request, this will return a Base64 version of `doc.to_bytes()` object.

you can recover it by following codes:

```python
from spacy.tokens import Doc
doc = nlp("Give it back! He pleaded.")
doc_bytes = doc.to_bytes()
doc2 = Doc(doc.vocab).from_bytes(doc_bytes)
assert doc.text == doc2.text
```

## use by whoosh query language

TODO

This can extract common nlp object for further investigation. 

there's some nlp object to query.

refer to https://whoosh.readthedocs.io/en/latest/querylang.html for more details.

type: doc, word, entity, closure(doubled_closure), chain(doubled_chain)
prop: text, pos, text_pos, depth, is_max_depth(is local maximum closure, chain or double-chain), is_double(allow/has two-direction edges flow to a center/ out a center), text_len, token_len, label(entity type), dep, pos_dep, text_pos_dep, span_start, span_end, word_id, word_ids

result may not contain all property to save network bandwidth (and increase search speed, index build time. if your query only need type=doc and text_len<5, then only type, text and text_len is returned.) This is not implemented.
