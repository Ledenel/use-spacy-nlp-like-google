from flask import Flask
app = Flask(__name__)
from flask import request
from flask import jsonify

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.filedb.filestore import RamStorage
from whoosh.analysis import KeywordAnalyzer
from whoosh.qparser import QueryParser

"""
type: doc, word, entity, closure(doubled_closure), chain(doubled_chain)
prop: 
    text,
    pos, 
    text_pos, 
    depth, 
    is_max_depth(is local maximum closure, chain or double-chain), 
    is_double(allow/has two-direction edges flow to a center/ out a center), 
    text_len, 
    token_len, 
    label(entity type), 
    dep, 
    pos_dep, 
    text_pos_dep, 
    span_start, 
    span_end, 
    head_word_id, 
    head_word_text,
    head_word_pos,
    word_ids
"""

word_info_schema = dict(
    type=ID(),
    text=TEXT(analyzer=KeywordAnalyzer()),
    text_len=NUMERIC(sortable=True),
    token_len=NUMERIC(sortable=True),
    pos=TEXT(), 
    pos_tag=TEXT(), 
    dep=TEXT(), 
    dep_edge=TEXT(),
    text_with_pos=TEXT(), 
    span_start=NUMERIC(sortable=True), 
    span_end=NUMERIC(sortable=True), 
    span_len=NUMERIC(sortable=True), 
    span_count=NUMERIC(sortable=True), 
    min_word_id=NUMERIC(sortable=True), 
    max_word_id=NUMERIC(sortable=True), 
    word_id=TEXT(),
    contains_ascii=BOOLEAN(),
    contains_punct=BOOLEAN(),
    is_max_depth=BOOLEAN(),
    depth=NUMERIC(sortable=True),
)

word_info_schema_numeric = {
    name:field for name,field in word_info_schema.items() if isinstance(field, NUMERIC)
}

schema = Schema(
    **word_info_schema,
    **{"head_%s" % k:v for k,v in word_info_schema.items()},
    **{"head_relative_%s" % k:v for k,v in word_info_schema_numeric.items()},
)

# from whoosh.qparser import QueryParser
# with ix.searcher() as searcher:
#     query = QueryParser("content", ix.schema).parse("first")
#     results = searcher.search(query)

from spacy.tokens import Doc, Token

def get_one_head(tokens, target:Token):
    while target != target.head and target.head in tokens:
        target = target.head
    return target

def get_head(tokens):
    heads = [get_one_head(tokens, tok) for tok in tokens]
    return min(heads, key=lambda token: token.i)

def get_words_info(_tokens):
    _token: Token
    _sorted_tokens = sorted(_tokens, key=lambda _token: _token.idx)
    text=[_token.text for _token in _sorted_tokens]
    text_len=sum(len(_x) for _x in text)
    token_len=len(text)
    text=" ".join(text)
    pos=" ".join([_token.pos_ for _token in _sorted_tokens])
    dep=" ".join([_token.dep_ for _token in _sorted_tokens])
    dep_edge= " ".join([str(_token.head.i) for _token in _sorted_tokens])
    text_with_pos = " ".join([_text for _token in _sorted_tokens for _text in [_token.text, _token.pos_]])
    pos_tag= " ".join([_token.tag_ for _token in _sorted_tokens])
    span_start = min(_token.idx for _token in _sorted_tokens)
    span_end = max(_token.idx + len(_token.text) for _token in _sorted_tokens)
    span_len = span_end - span_start
    _span_changed = (right is not None and left.idx+len(left.text) != right.idx for left, right in zip(_sorted_tokens, _sorted_tokens[1:] + [None]))
    span_count = sum(_span_changed) + 1
    min_word_id = min(_token.i for _token in _sorted_tokens)
    max_word_id = max(_token.i for _token in _sorted_tokens)
    word_id = " ".join([str(_token.i) for _token in _sorted_tokens])
    contains_ascii = any(ch.isascii() for _token in _sorted_tokens for ch in _token.text)
    contains_punct = any(_token.is_punct for _token in _sorted_tokens)
    return {k:v for k,v in locals().items() if not k.startswith("_")}

def get_composed_info(tokens, head=None):
    word_infos = get_words_info(tokens)
    if head is None:
        head = get_head(tokens)
    head_info = get_words_info([head])
    return {
        **word_infos,
        **{"head_%s" % k:v for k,v in head_info.items()},
        **{"head_relative_%s" % k:head - word_infos[k] for k,head in head_info.items() if k in word_info_schema_numeric},
    }

def collect_closure(head: Token):
    closure = set()
    closure.add(head)
    doc = head.doc
    has_more = True
    while has_more:
        has_more = False
        yield closure.copy()
        for token in doc:
            if token.head in closure and token not in closure:
                closure.add(token)
                has_more = True
            

from collections import defaultdict
import spacy

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None or not callable(self.default_factory):
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret

model_dict = keydefaultdict(spacy.load)
model_dict["zh_core_web_sm"]

from tqdm import tqdm
from time import perf_counter

def get_all_fields(query):
    if query.field() is not None:
        yield query.field()
    for child in query.children():
        yield from get_all_fields(child)

from functools import total_ordering

import networkx as nx

@total_ordering
class SearchingItem:
    def __init__(self, graph, head, depth, children, search_list, unsearched_list, attr_cache, fields):
        self.graph = graph
        self.head = head
        self.depth = depth
        self.children = list(children)
        self.children_key = frozenset(t.i for t in children)
        self.search_list = search_list
        self.unsearched_list = unsearched_list
        self.attr_cache = attr_cache
        self.fields = fields
    
    @classmethod
    def new_search(cls, graph: nx.DiGraph, fields = {
        "search_len": True,
        "span_count": True,
        "token_len": False,
    }):
        attr_cache={}
        for node in graph.nodes:
            yield SearchingItem(
                graph=graph,
                head=node,
                depth=1,
                children=[node],
                search_list=[(2, n) for n in graph.adj[node]],
                unsearched_list=[],
                attr_cache=attr_cache,
                fields=fields,
            )

    def expand(self):
        if self.search_list:
            first, *rest = self.search_list
            searching_depth, first_node = first
            yield SearchingItem(
                self.graph, 
                self.head,
                max(self.depth, searching_depth),
                self.children + [first_node], 
                rest + [(searching_depth + 1, n) for n in self.graph.adj[first_node]],
                self.unsearched_list, 
                self.attr_cache, 
                self.fields,
            )
            yield SearchingItem(
                self.graph, 
                self.head, 
                self.depth,
                self.children, 
                rest, 
                self.unsearched_list + [first], 
                self.attr_cache, 
                self.fields,
            )

    @property
    def attr(self):
        result = self.attr_cache.get(self.children_key, None)
        if result is None:
            result = get_composed_info(self.children, self.head)
            result["search_len"] = len(self.search_list)
            self.attr_cache[self.children_key] = result
        return result
    
    def __lt__(self, other):
        for field, ascending in self.fields.items():
            if self.attr[field] == other.attr[field]:
                pass
            else:
                return self.attr[field] < other.attr[field] if ascending else self.attr[field] > other.attr[field]
        return False

    def __eq__(self, other):
        return [self.attr[field] for field in self.fields] == [other.attr[field] for field in self.fields]

from queue import PriorityQueue

def search_subtrees(doc):
    graph = nx.DiGraph()
    for token in doc:
        if token.head != token:
            graph.add_edge(token.head, token)
    queue = PriorityQueue()
    for item in SearchingItem.new_search(graph):
        queue.put(item, block=False)
    while not queue.empty():
        item = queue.get(block=False)
        if not item.search_list:
            yield item
        for sub_item in item.expand():
            queue.put(sub_item, block=False)
        
    

@app.route('/', methods=['GET'])
def nlp():
    _st = perf_counter()
    params = {**dict(
        model="zh_core_web_sm",
        type="doc,word",
        query="type:doc",
        text="假如你是李华，你打算给Tom写一封信，信中描述你在中国的生活。要求不少于200词。",
        text_only="no",
        word_mode="no",
        subtree_limit="500",    
    ), **request.args, **request.form
    }
    params.setdefault("limit", len(params["text"]))
    params["subtree_limit"] = int(params["subtree_limit"])
    params["text_only"] = params["text_only"] == "yes"
    params["word_mode"] = params["word_mode"] == "yes"
    print("request done", params["query"], perf_counter() - _st); _st = perf_counter()
    doc = model_dict[params["model"]](params["text"])
    print("nlp parse done, to write", perf_counter() - _st); _st = perf_counter()
    search_items = []
    search_items.append(dict(type="doc", **get_composed_info(list(doc))))
    for token in doc:
        search_items.append(dict(type="word", **get_composed_info([token])))
        # closures = list(enumerate(collect_closure(token)))
        # is_max_depth_items = [False for _ in closures]
        # is_max_depth_items[-1] = True
        # for (depth, closure), max_depth in zip(closures, is_max_depth_items):
        #     search_items.append(dict(
        #         type="closure",
        #         **get_composed_info(list(closure)),
        #         is_max_depth=max_depth,
        #         depth=depth,
        #     ))
    print("adding subtrees", perf_counter() - _st); _st = perf_counter()
    for subtree_item, _ in zip(search_subtrees(doc), range(500)):
        search_items.append(dict(
            type="subtree",
            **subtree_item.attr,
            is_max_depth=not subtree_item.unsearched_list,
            depth=subtree_item.depth
        ))
    print("search items done", perf_counter() - _st); _st = perf_counter()
    pre_query = QueryParser("text", schema).parse(params["query"])
    fields = list(get_all_fields(pre_query))
    final_schema = {k:schema[k] for k in fields}
    final_schema["id"] = STORED()
    final_schema = Schema(**final_schema)
    print(final_schema)
    ix = RamStorage().create_index(final_schema)
    print("create schema", perf_counter() - _st); _st = perf_counter()
    writer = ix.writer()
    print("create writer", perf_counter() - _st); _st = perf_counter()
    for i, item in enumerate(search_items):
        writer.add_document(id=i, **{k:item[k] for k in fields if k in item})
    print("add document to memory", perf_counter() - _st); _st = perf_counter()
    writer.commit()
    print("commited", perf_counter() - _st); _st = perf_counter()
    print("now doc", ix.doc_count_all())
    
    with ix.searcher() as searcher:
        query = QueryParser("text", ix.schema).parse(params["query"])
        results = searcher.search(query, limit=params["limit"])
        print("search done", perf_counter() - _st); _st = perf_counter()
        search_items_result = [search_items[hit["id"]] for hit in results]
        if params["text_only"]:
            search_items_result = [item["text"] for item in search_items_result]
            if params["word_mode"]:
                search_items_result = [t.split(" ") for t in search_items_result]
            else:
                search_items_result = [t.replace(" ", "") for t in search_items_result]
        return jsonify(search_items_result)





import os
if "FLASK_DEBUG" in os.environ and str(os.environ["FLASK_DEBUG"]) == "1":
    pass
else:
    @app.errorhandler(Exception)
    def exception_handler(e):
        return str(e), 400
