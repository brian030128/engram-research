"""Run the unmodified official demo as __main__, but resolve the tokenizer to a local
tokenizer-only snapshot with trust_remote_code=False (no repo code gets executed)."""
import runpy,sys,time
import transformers
LOCAL=sys.argv[1]; DEMO=sys.argv[2]
_orig=transformers.AutoTokenizer.from_pretrained
def patched(name,*a,**k):
    k['trust_remote_code']=False
    return _orig(LOCAL,*a,**k)
transformers.AutoTokenizer.from_pretrained=patched
t=time.time(); runpy.run_path(DEMO,run_name='__main__'); print(f'demo_wall_s={time.time()-t:.1f}')
