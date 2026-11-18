"""Run after building A and B. Optionally pass the hidden-label A output."""
import argparse
import math
from pathlib import Path
from pipeline import read, FIELDS, LEVELS

def check(root, name):
    folder = root / ("company_" + name)
    sources = ([("erp_goods_receipts.csv","ERP:","line_id"),
                ("invoice_tool_export.csv","TOOL:","tool_record_id")] if name=="A"
               else [("invoice_lines.csv","","line_id")])
    ids = [prefix+r[col] for file,prefix,col in sources for r in read(folder/file)]
    rows = read(Path(f"Abhishek_Singh__{name}.csv"))
    paths = {tuple(r.values())+("",) for r in read(folder/f"taxonomy_{name}.csv")}
    assert list(rows[0]) == FIELDS
    assert ids == [r["record_id"] for r in rows] and len(ids)==len(set(ids))
    for r in rows:
        assert r["include"] in {"0","1"}
        assert 0<=float(r["confidence"])<=1 and math.isfinite(float(r["spend"]))
        assert r["item_key"] and r["supplier_key"]
        if r["include"]=="1":
            assert not r["exclusion_reason"] and tuple(r[k] for k in LEVELS) in paths
        else:
            assert float(r["spend"])==0 and r["exclusion_reason"] in {"DUPLICATE","CANCELLED","NOT_APPROVED","NON_PROCUREMENT"}
    print(name,len(rows),"records verified")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kit",type=Path)
    parser.add_argument("--hidden",type=Path)
    args = parser.parse_args()
    for company in ("A","B"):
        check(args.kit,company)
    if args.hidden:
        predictions = {r["record_id"]:r for r in read(args.hidden)}
        labels = read(args.kit/"company_A"/"labels_A.csv")
        for k in LEVELS[:3]:
            hits = sum(predictions[r["record_id"]][k]==r["category_"+k] for r in labels)
            print(k,hits,len(labels),f"{hits/len(labels):.2%}")
