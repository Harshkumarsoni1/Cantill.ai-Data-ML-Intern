"""Config-driven offline spend classification. See README for the CLI."""
import argparse
import csv
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


LEVELS = ["l1", "l2", "l3", "l4"]
FIELDS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend", *LEVELS, "confidence"]
EXCLUDE = r"\b(intercompany|inter company|i c payable|ic payables?|ic\d{2}|sales tax|income tax|withholding tax|vat payable|royalt(?:y|ies)|partner billings?|financ(?:ing|e) (?:charges?|costs?)|interest expense|revenue)\b"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=150000)
def clean(value):
    text = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(re.findall(r"[^\W_]+", "".join(c for c in text if not unicodedata.combining(c))))


@lru_cache(maxsize=150000)
def key(value):
    return hashlib.sha256(value.encode()).hexdigest()[:18]


def number(value, decimal="."):
    value = value.strip().replace(" ", "").replace("\u00a0", "").strip("%")
    if decimal == ",":
        value = value.replace(".", "").replace(",", ".")
    else:
        value = value.replace(",", "")
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except ValueError:
        return None


@lru_cache(maxsize=12000)
def month(value, formats):
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y-%m")
        except ValueError:
            pass
    return ""


def load_rows(cfg, folder, audit):
    rates = {}
    if cfg.get("fx") and (folder / cfg["fx"]).exists():
        rates = {(r["currency"], r["month"]): number(r["units_per_usd"]) for r in read(folder / cfg["fx"])}
    result = []
    for src in cfg["sources"]:
        rows = read(folder / src["file"])
        columns = [src[k] for k in ("id","supplier","item","amount","tax","date","po","quantity","invoice","status","account","currency","rate","fallback_date") if k in src]
        columns += src["description"] + src.get("context", [])
        if rows and (missing := set(columns) - rows[0].keys()):
            raise ValueError(f"Missing configured columns in {src['file']}: {sorted(missing)}")
        for raw in rows:
            flags = []
            def field(k):
                return (raw.get(src.get(k), "") or "").strip()
            rid = src.get("prefix", "") + field("id")
            text = clean(" ".join(raw[c] for c in src["description"]))
            context = clean(" ".join(raw[c] for c in src.get("context", [])))
            account = clean(field("account"))
            supplier = re.sub(r"\b(pty|ltd|limited|inc|pvt|llc|plc)\b", " ", clean(field("supplier")))
            supplier = " ".join(supplier.split()) or rid
            status = clean(field("status"))
            reason = ""
            if "status" in src and status not in src.get("approved", ["paid","approved"]):
                reason = "CANCELLED" if "cancel" in status or "void" in status else "NOT_APPROVED"
            elif re.search(cfg.get("non_procurement", EXCLUDE), account + " " + text):
                reason = "NON_PROCUREMENT"
            formats = tuple(src.get("date_formats", ["%Y-%m-%d","%d/%m/%Y","%d-%b-%y","%m/%d/%Y","%d.%m.%Y","%Y/%m/%d"]))
            period = month(field("date"), formats)
            if not period:
                flags.append("blank_or_invalid_date")
                audit["blank_or_invalid_date"] += 1
                period = month(field("fallback_date"), formats)
            amount = number(field("amount"), src.get("decimal","."))
            if amount is None:
                flags.append("invalid_amount")
                audit["invalid_amount"] += 1
                amount = 0
            if "tax" in src:
                tax = number(field("tax"))
                if tax is None:
                    tax = src.get("tax_default", 0)
                elif tax > 1 or "%" in field("tax"):
                    tax /= 100
                amount /= 1 + tax
            currency = field("currency") or cfg["currency"]
            if currency != cfg["currency"]:
                rate = number(field("rate"))
                if not rate or rate <= 0:
                    rate = rates.get((currency, period))
                if rate and rate > 0:
                    amount /= rate
                else:
                    flags.append("missing_fx")
                    audit["missing_fx"] += 1
                    amount = 0
            po = re.search(r"\d{5,}", field("po"))
            result.append(dict(id=rid, text=text, context=context, supplier=supplier, code=clean(field("item")),
                               amount=amount, reason=reason, role=src.get("role",""), po=po[0] if po else "",
                               qty=number(field("quantity")), invoice=clean(field("invoice")), account=account, flags=flags))
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Input record IDs must be unique")
    return result


def reconcile(rows):
    by_po, by_amount = defaultdict(list), defaultdict(list)
    seen, used = set(), set()
    for r in rows:
        if r["role"] == "primary":
            by_po[r["po"]].append(r)
            by_amount[round(r["amount"])].append(r)
        if r["role"] != "secondary" or r["reason"]:
            continue
        sig = (r["invoice"],r["supplier"],r["text"],r["qty"],round(r["amount"],2))
        if sig in seen:
            r["reason"] = "DUPLICATE"
            continue
        seen.add(sig)
        pool = by_po[r["po"]] if r["po"] else []
        if not pool:
            pool = by_amount[round(r["amount"])]
        best, high = None, 0
        for e in pool:
            if e["id"] in used:
                continue
            same_po = bool(r["po"]) and e["po"] == r["po"]
            close = abs(e["amount"] - r["amount"]) <= max(.015, abs(e["amount"])*.0001)
            if not (same_po or close):
                continue
            sim = SequenceMatcher(None,r["text"],e["text"]).ratio()
            score = 2*same_po + 3*close + 2*sim + (r["supplier"]==e["supplier"]) + .5*(r["qty"]==e["qty"])
            if score > high and (close and (same_po or sim>.63) or same_po and sim>.92 and r["qty"]==e["qty"]):
                best, high = e, score
        if best:
            used.add(best["id"])
            r["reason"] = "DUPLICATE"
            r["linked"] = best


def features(r):
    return " ".join([r["text"],r["text"],r["context"],"vendor"+key(r["supplier"]), "mat"+r["code"].replace(" ","")])


def classify(rows, paths, training, labels, levels):
    indices = {p:i for i,p in enumerate(paths)}
    training = [(r,indices[p]) for r in training if (lab:=labels.get(r["id"])) and
                (p:=tuple(lab.get(k,"") for k in levels)) in indices and lab.get("include","1")=="1"]
    if len(set(y for _,y in training)) >= 2:
        vec = FeatureUnion([("words",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
                            ("chars",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,4),min_df=2,sublinear_tf=True))])
        x = vec.fit_transform([features(r) for r,_ in training])
        model = LinearSVC(random_state=0).fit(x,[y for _,y in training])
        for start in range(0,len(rows),2000):
            batch = rows[start:start+2000]
            scores = model.decision_function(vec.transform([features(r) for r in batch]))
            if scores.ndim == 1:
                scores = np.column_stack((-scores,scores))
            for r,s in zip(batch,scores):
                ranked = np.argsort(s)
                r["path"] = paths[model.classes_[ranked[-1]]]
                r["confidence"] = 1/(1+math.exp(-min(20,s[ranked[-1]]-s[ranked[-2]])))
    else:
        docs = [" ".join(p) + " " + next((v for v in reversed(p) if v),"") for p in paths]
        v = TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),strip_accents="unicode")
        tree = v.fit_transform(docs)
        # Use the category scores of other purchases from the same supplier as
        # weak context. Repeated descriptions receive one vote per supplier.
        evidence = defaultdict(lambda: np.zeros(len(paths)))
        counts = Counter()
        seen = set()
        for start in range(0,len(rows),2000):
            batch = rows[start:start+2000]
            scores = (v.transform([r["text"]+" "+r["context"] for r in batch]) @ tree.T).toarray()
            for r,s in zip(batch,scores):
                r["scores"] = s
                bucket = (r["supplier"], r["text"])
                if bucket not in seen and r["text"] and s.max()>0:
                    evidence[r["supplier"]] += s
                    counts[r["supplier"]] += 1
                    seen.add(bucket)
        for r in rows:
            n = counts[r["supplier"]]
            scores = r.pop("scores")
            if n >= 3:
                scores = .72*scores + .28*evidence[r["supplier"]]/n
            r["path"] = paths[int(scores.argmax())]
            r["confidence"] = float(scores.max())


def run(cfgfile, data, reference, out, hidden=False):
    start = time.monotonic()
    cfg = json.loads(Path(cfgfile).read_text())
    audit = Counter()
    rows = load_rows(cfg,data,audit)
    tree = data / cfg["taxonomy"]
    levels = cfg["levels"]
    paths = list(dict.fromkeys(tuple(r[k] for k in levels) for r in read(tree if tree.exists() else reference/tree.name)))
    labfile = reference / cfg.get("labels","_none_")
    labels = {} if hidden or not labfile.is_file() else {r[cfg["label_id"]]:r for r in read(labfile)}
    train = rows if reference.resolve()==data.resolve() else load_rows(cfg,reference,Counter()) if labels else []
    classify(rows,paths,train,labels,levels)
    suppliers, items, descriptions, excludes = (defaultdict(Counter) for _ in range(4))
    for r in train:
        lab = labels.get(r["id"],{})
        if lab.get("supplier_key"):
            suppliers[r["supplier"]][lab["supplier_key"]] += 1
        if lab.get("item_key") and r["code"]:
            items[r["code"]][lab["item_key"]] += 1
        if r["code"] and r["text"]:
            descriptions[r["text"]][r["code"]] += 1
        if r["text"] and lab.get("include"):
            excludes[r["text"]][lab["exclusion_reason"]] += 1
    for r in rows:
        votes = excludes[r["text"]]
        if not r["reason"] and votes and sum(votes.values())>=2 and len(votes)==1:
            r["reason"] = next(iter(votes))
    reconcile(rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for r in rows:
            entity = r.get("linked",r)
            supplier = entity["supplier"]
            code = entity["code"]
            guesses = descriptions[entity["text"]]
            if not code and len(guesses)==1:
                code = next(iter(guesses))
            item = items[code].most_common(1)[0][0] if items[code] else key(code or entity["text"] or entity["id"])
            skey = suppliers[supplier].most_common(1)[0][0] if suppliers[supplier] else key(supplier)
            inc = not r["reason"]
            path = list(r["path"]) if inc else []
            writer.writerow([r["id"],int(inc),r["reason"],skey,item,format(r["amount"] if inc else 0,".2f"),*(path+[""]*(4-len(path))),round(r["confidence"],4) if inc else 0])
            audit[r["reason"] or "included"]+=1
    audit_dir = Path("local_audit")
    audit_dir.mkdir(exist_ok=True)
    with (audit_dir / out.name).open("w",newline="",encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id","flags"])
        writer.writerows((r["id"],"|".join(r["flags"])) for r in rows if r["flags"])
    print(out.name,len(rows),dict(audit),f"{time.monotonic()-start:.1f}s",flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--kit",type=Path,help="Run both build companies from the kit root")
    cli.add_argument("--config",type=Path)
    cli.add_argument("--data",type=Path)
    cli.add_argument("--reference",type=Path)
    cli.add_argument("--out",type=Path)
    cli.add_argument("--hide-labels",action="store_true")
    args = cli.parse_args()
    if args.kit:
        for company in ("A","B"):
            folder = args.kit / f"company_{company}"
            run(f"configs/{company}.json",folder,folder,Path(f"Abhishek_Singh__{company}.csv"),args.hide_labels)
    elif args.config and args.data and args.out:
        run(args.config,args.data,args.reference or args.data,args.out,args.hide_labels)
    else:
        cli.error("Supply --kit or --config, --data and --out")
