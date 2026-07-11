# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json, sys
def assemble(percase):
    # percase: list of {deltas:[{index,name?,arguments?}], normal_text}
    order=[]; names={}; args={}; normal=""
    for ch in percase:
        normal += ch.get("normal_text","")
        for d in ch.get("deltas",[]):
            i=d["index"]
            if i not in order: order.append(i)
            if "name" in d: names[i]=names.get(i,"")+d["name"]
            if "arguments" in d: args[i]=args.get(i,"")+d["arguments"]
    calls=[]
    for i in order:
        raw=args.get(i,"")
        try: a=json.loads(raw) if raw else {}
        except Exception: a=raw
        calls.append({"name":names.get(i,""),"arguments":a})
    return {"calls":calls,"normal_text":normal}
src=json.load(open(sys.argv[1]))
out={fam:{cid:assemble(pc) for cid,pc in cases.items()} for fam,cases in src.items()}
json.dump(out, open(sys.argv[2],"w"))
print(f"assembled {sum(len(c) for c in out.values())} cases -> {sys.argv[2]}")
