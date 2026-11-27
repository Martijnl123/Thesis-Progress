# Leakage test (review #107): relabel the 5,686 borrower names with an OPEN-WEIGHTS model whose published training cutoff is
# before the 2025-26 software sell-off, and compare with the Claude labels of 26 Oct 2026. Runs on a local PC with Ollama.
# Written 17 Nov 2026. Reads only industry_llm/full/run_input.csv and prompt.md (borrower names; no marks, no dates after 2025).
#
# Model choice (published cutoffs, checked 17 Nov 2026 on the model cards):
#   gpt-oss:20b   OpenAI open weights, "Jun 01, 2024 knowledge cutoff" (developers.openai.com model page); runs in 16 GB RAM/VRAM.
#   llama4:scout  Meta Llama 4 Scout, "pretraining data has a cutoff of August 2024" (model card); needs about 64 GB+ of memory.
# Use llama4:scout if the PC can run it (later cutoff, recognises more names); otherwise gpt-oss:20b.
#
# Setup (Windows): install Ollama from ollama.com, then in a terminal:
#   ollama pull gpt-oss:20b        (or: ollama pull llama4:scout)
#   python open_model_relabel_pc.py --model gpt-oss:20b
# It resumes where it stopped. Output: industry_llm/open-model-2026-11-17/out_<model>.csv plus a run log with the model digest.
import argparse, csv, io, json, os, sys, time, urllib.request
ap=argparse.ArgumentParser(); ap.add_argument('--model',default='gpt-oss:20b'); ap.add_argument('--batch',type=int,default=20)
ap.add_argument('--root',default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','industry_llm'))
ap.add_argument('--host',default='http://localhost:11434')
ap.add_argument('--input',default='run_input.csv',help='file in industry_llm/full; run_input_pilot200.csv for the pilot'); a=ap.parse_args()
FULL=os.path.join(a.root,'full'); OUT=os.path.join(a.root,'open-model-2026-11-21'); os.makedirs(OUT,exist_ok=True)
tag=a.model.replace(':','_').replace('/','_')+('' if a.input=='run_input.csv' else '_'+a.input.replace('.csv','')); outf=os.path.join(OUT,f'out_{tag}.csv'); logf=os.path.join(OUT,f'log_{tag}.txt'); rawf=os.path.join(OUT,f'raw_{tag}.txt')
prompt=open(os.path.join(FULL,'prompt.md'),encoding='utf-8').read()
prompt=prompt.replace('Write a CSV file with exactly this header and one row per input item, in the input order:','Answer with CSV only: exactly this header and one row per input item, in the input order:')
prompt=prompt.replace('Do not use web search or any other tool except reading the input file and writing the output file.','Do not use any tool.')
rows=list(csv.DictReader(open(os.path.join(FULL,a.input),encoding='utf-8')))
done=set()
if os.path.exists(outf):
    done={r['item'] for r in csv.DictReader(open(outf,encoding='utf-8'))}
HDR=['item','operating_name','group','software','it_services','ai_exposure','known']
def call(batch):
    buf=io.StringIO(); w=csv.writer(buf); w.writerow(['item','text']); [w.writerow([r['item'],r['text']]) for r in batch]
    body={'model':a.model,'stream':False,'options':{'temperature':0,'seed':20261117,'num_ctx':16384,'num_predict':12000},
          'messages':[{'role':'system','content':prompt},{'role':'user','content':'Input file:\n'+buf.getvalue()}]}
    req=urllib.request.Request(a.host+'/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    m=json.loads(urllib.request.urlopen(req,timeout=1800).read())
    txt=m['message']['content']
    with open(rawf,'a',encoding='utf-8') as R: R.write(f"=== items {batch[0]['item']}-{batch[-1]['item']} done_reason={m.get('done_reason')} eval_count={m.get('eval_count')} prompt_eval_count={m.get('prompt_eval_count')}\n{txt}\n")
    return txt
def parse(txt,batch):
    txt=txt.replace('```csv','').replace('```','').strip(); i=txt.find('item,operating_name'); txt=txt[i:] if i>=0 else txt
    got={}
    for r in csv.DictReader(io.StringIO(txt)):
        if r.get('item') and r.get('group'): got[r['item'].strip()]={k:(r.get(k) or '').strip() for k in HDR}
    return got
info=json.loads(urllib.request.urlopen(a.host+'/api/show',data=json.dumps({'model':a.model}).encode()).read())
with open(logf,'a',encoding='utf-8') as L: L.write(f'{time.strftime("%Y-%m-%d %H:%M")} model {a.model} details {json.dumps(info.get("details",{}))} modified {info.get("modified_at","")} temperature 0 seed 20261117 num_ctx 16384\n')
new=not os.path.exists(outf)
with open(outf,'a',newline='',encoding='utf-8') as fo:
    w=csv.DictWriter(fo,fieldnames=HDR)
    if new: w.writeheader()
    todo=[r for r in rows if r['item'] not in done]; print(f'{len(todo)} of {len(rows)} items to label with {a.model}')
    for k in range(0,len(todo),a.batch):
        b=todo[k:k+a.batch]
        for attempt in range(3):
            try: got=parse(call(b),b); break
            except Exception as e: print('retry',attempt,e); time.sleep(5); got={}
        for r in b:
            g=got.get(r['item'])
            if g: w.writerow(g)
        fo.flush(); print(f'{min(k+a.batch,len(todo))}/{len(todo)} done; this batch parsed {len(got)} of {len(b)}',flush=True)
print('finished; rerun the same command to fill any items that failed to parse')
